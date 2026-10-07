import { Canvas, useFrame } from "@react-three/fiber";
import { ContactShadows, Float, OrbitControls } from "@react-three/drei";
import { Suspense, useMemo, useRef } from "react";
import * as THREE from "three";
import type { FingerPose } from "../../services/types";

/* ------------------------------------------------------------------ */
/* Materials                                                           */
/* ------------------------------------------------------------------ */

const useMaterials = () =>
  useMemo(
    () => ({
      shell: new THREE.MeshPhysicalMaterial({
        color: "#0b0d12",
        metalness: 0.55,
        roughness: 0.32,
        clearcoat: 1,
        clearcoatRoughness: 0.15,
      }),
      joint: new THREE.MeshStandardMaterial({ color: "#232a38", metalness: 0.7, roughness: 0.4 }),
      accent: new THREE.MeshStandardMaterial({
        color: "#22d3ee",
        emissive: "#22d3ee",
        emissiveIntensity: 1.6,
        toneMapped: false,
      }),
      accentDim: new THREE.MeshStandardMaterial({
        color: "#8b5cf6",
        emissive: "#8b5cf6",
        emissiveIntensity: 0.8,
        toneMapped: false,
      }),
    }),
    [],
  );

type Mats = ReturnType<typeof useMaterials>;

/* ------------------------------------------------------------------ */
/* Finger                                                              */
/* ------------------------------------------------------------------ */

interface FingerProps {
  position: [number, number, number];
  rotation?: [number, number, number];
  lengths: [number, number, number];
  radius: number;
  curlRef: React.MutableRefObject<number>;
  mats: Mats;
  isThumb?: boolean;
}

function Segment({ length, radius, mats, tip }: { length: number; radius: number; mats: Mats; tip?: boolean }) {
  return (
    <group>
      {/* joint knuckle */}
      <mesh material={mats.joint} rotation={[0, 0, Math.PI / 2]}>
        <cylinderGeometry args={[radius * 1.05, radius * 1.05, radius * 1.6, 20]} />
      </mesh>
      {/* phalanx shell */}
      <mesh material={mats.shell} position={[0, length / 2, 0]}>
        <capsuleGeometry args={[radius * 0.92, length - radius * 0.8, 6, 16]} />
      </mesh>
      {/* accent strip */}
      <mesh material={mats.accent} position={[0, length / 2, radius * 0.86]}>
        <boxGeometry args={[radius * 0.5, length * 0.5, radius * 0.12]} />
      </mesh>
      {tip && (
        <mesh material={mats.joint} position={[0, length, 0]}>
          <sphereGeometry args={[radius * 0.85, 16, 16]} />
        </mesh>
      )}
    </group>
  );
}

function Finger({ position, rotation = [0, 0, 0], lengths, radius, curlRef, mats, isThumb }: FingerProps) {
  const j1 = useRef<THREE.Group>(null);
  const j2 = useRef<THREE.Group>(null);
  const j3 = useRef<THREE.Group>(null);
  const current = useRef(0);

  useFrame((_, dt) => {
    const target = curlRef.current;
    current.current += (target - current.current) * Math.min(1, dt * 7);
    const c = current.current;
    const max = isThumb ? 0.95 : 1.55;
    if (j1.current) j1.current.rotation.x = -c * max * 0.75;
    if (j2.current) j2.current.rotation.x = -c * max * 0.9;
    if (j3.current) j3.current.rotation.x = -c * max * 0.6;
  });

  return (
    <group position={position} rotation={rotation}>
      <group ref={j1}>
        <Segment length={lengths[0]} radius={radius} mats={mats} />
        <group ref={j2} position={[0, lengths[0], 0]}>
          <Segment length={lengths[1]} radius={radius * 0.92} mats={mats} />
          <group ref={j3} position={[0, lengths[1], 0]}>
            <Segment length={lengths[2]} radius={radius * 0.82} mats={mats} tip />
          </group>
        </group>
      </group>
    </group>
  );
}

/** Thumb base orientation (Euler XYZ) at rest, and when its tip touches the index tip (solved numerically). */
const THUMB_REST: [number, number, number] = [0.35, 0.3, 1.05];
const THUMB_TOUCH: [number, number, number] = [-0.782, 0.13, -0.116];

/* ------------------------------------------------------------------ */
/* Hand assembly                                                       */
/* ------------------------------------------------------------------ */

function HandModel({ poseRef }: { poseRef: React.RefObject<FingerPose> }) {
  const mats = useMaterials();
  const thumb = useRef(0);
  const index = useRef(0);
  const middle = useRef(0);
  const ring = useRef(0);
  const pinky = useRef(0);
  const thumbBase = useRef<THREE.Group>(null);
  const opp = useRef(0);
  const root = useRef<THREE.Group>(null);
  const ringLight = useRef<THREE.Mesh>(null);

  useFrame(({ clock }, dt) => {
    const p = poseRef.current ?? [0, 0, 0, 0, 0, 0];
    thumb.current = p[0];
    index.current = p[1];
    middle.current = p[2];
    ring.current = p[3];
    pinky.current = p[4];
    // Thumb opposition: swing the thumb from its rest orientation to the solved
    // "tip-touches-index-tip" orientation (smoothed).
    opp.current += ((p[5] ?? 0) - opp.current) * Math.min(1, dt * 7);
    if (thumbBase.current) {
      const o = opp.current;
      thumbBase.current.rotation.set(
        THUMB_REST[0] + (THUMB_TOUCH[0] - THUMB_REST[0]) * o,
        THUMB_REST[1] + (THUMB_TOUCH[1] - THUMB_REST[1]) * o,
        THUMB_REST[2] + (THUMB_TOUCH[2] - THUMB_REST[2]) * o,
      );
    }
    if (ringLight.current) {
      const m = ringLight.current.material as THREE.MeshStandardMaterial;
      m.emissiveIntensity = 1.2 + Math.sin(clock.elapsedTime * 3) * 0.5;
    }
  });

  const palmW = 1.55;
  const palmH = 1.7;

  return (
    <group ref={root} position={[0, -0.9, 0]}>
      {/* Palm */}
      <mesh material={mats.shell} position={[0, palmH / 2, 0]}>
        <boxGeometry args={[palmW, palmH, 0.55, 4, 4, 2]} />
      </mesh>
      {/* Palm plate (back) */}
      <mesh material={mats.joint} position={[0, palmH / 2 + 0.05, -0.29]}>
        <boxGeometry args={[palmW * 0.78, palmH * 0.7, 0.06]} />
      </mesh>
      {/* Palm accent core */}
      <mesh ref={ringLight} material={mats.accent} position={[0, palmH * 0.55, -0.33]}>
        <torusGeometry args={[0.22, 0.03, 12, 40]} />
      </mesh>
      <mesh material={mats.accentDim} position={[0, palmH * 0.55, -0.33]}>
        <circleGeometry args={[0.12, 24]} />
      </mesh>
      {/* Knuckle bar */}
      <mesh material={mats.joint} position={[0, palmH, 0]} rotation={[0, 0, Math.PI / 2]}>
        <cylinderGeometry args={[0.2, 0.2, palmW * 0.98, 24]} />
      </mesh>

      {/* Wrist */}
      <mesh material={mats.joint} position={[0, -0.15, 0]}>
        <cylinderGeometry args={[0.52, 0.6, 0.35, 32]} />
      </mesh>
      <mesh material={mats.accent} position={[0, -0.05, 0]} rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[0.56, 0.02, 10, 48]} />
      </mesh>
      <mesh material={mats.shell} position={[0, -0.95, 0]}>
        <cylinderGeometry args={[0.48, 0.55, 1.3, 32]} />
      </mesh>
      {/* Forearm vents */}
      {[-0.5, -0.75, -1.0, -1.25].map((y) => (
        <mesh key={y} material={mats.joint} position={[0, y, 0]} rotation={[Math.PI / 2, 0, 0]}>
          <torusGeometry args={[0.5, 0.012, 8, 48]} />
        </mesh>
      ))}

      {/* Fingers: index, middle, ring, pinky */}
      <Finger position={[-0.58, palmH, 0]} rotation={[0, 0, 0.06]} lengths={[0.62, 0.42, 0.32]} radius={0.15} curlRef={index} mats={mats} />
      <Finger position={[-0.19, palmH + 0.08, 0]} lengths={[0.68, 0.46, 0.34]} radius={0.155} curlRef={middle} mats={mats} />
      <Finger position={[0.2, palmH + 0.03, 0]} rotation={[0, 0, -0.04]} lengths={[0.62, 0.42, 0.32]} radius={0.148} curlRef={ring} mats={mats} />
      <Finger position={[0.58, palmH - 0.12, 0]} rotation={[0, 0, -0.12]} lengths={[0.48, 0.34, 0.27]} radius={0.13} curlRef={pinky} mats={mats} />
      {/* Thumb */}
      <group ref={thumbBase} position={[-0.778, 0.768, 0.077]} rotation={THUMB_REST}>
        <mesh material={mats.joint}>
          <sphereGeometry args={[0.2, 20, 20]} />
        </mesh>
        <Finger position={[0, 0, 0]} lengths={[0.6, 0.48, 0.36]} radius={0.17} curlRef={thumb} mats={mats} isThumb />
      </group>
    </group>
  );
}

/* ------------------------------------------------------------------ */
/* Scene wrapper                                                       */
/* ------------------------------------------------------------------ */

export function RoboticHand({ poseRef, interactive = true, className }: { poseRef: React.RefObject<FingerPose>; interactive?: boolean; className?: string }) {
  return (
    <div className={className ?? "h-full w-full"}>
      <Canvas camera={{ position: [0, 1.2, 6.2], fov: 38 }} dpr={[1, 1.8]} gl={{ antialias: true, alpha: true }}>
        <Suspense fallback={null}>
          <ambientLight intensity={0.35} />
          <spotLight position={[4, 6, 5]} angle={0.4} penumbra={0.8} intensity={60} color="#9bdcff" />
          <pointLight position={[-4, 2, 3]} intensity={25} color="#8b5cf6" />
          <pointLight position={[0, -2, -4]} intensity={15} color="#22d3ee" />
          <directionalLight position={[-3, 4, -4]} intensity={2.5} color="#67e8f9" />
          <hemisphereLight args={["#334155", "#000000", 0.6]} />
          <Float speed={1.4} rotationIntensity={0.25} floatIntensity={0.4}>
            <group scale={0.74} position={[0, -0.25, 0]}>
              <HandModel poseRef={poseRef} />
            </group>
          </Float>
          <ContactShadows position={[0, -2.3, 0]} opacity={0.6} scale={8} blur={2.5} far={4} color="#000" />
          {interactive && (
            <OrbitControls enablePan={false} enableZoom={false} minPolarAngle={Math.PI / 3} maxPolarAngle={(Math.PI * 2) / 3} autoRotate autoRotateSpeed={0.6} />
          )}
        </Suspense>
      </Canvas>
    </div>
  );
}
