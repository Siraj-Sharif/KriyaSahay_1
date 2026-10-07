import { AnimatePresence, motion } from "framer-motion";
import { useCallback, useEffect, useLayoutEffect, useRef, useState, type ReactNode, type RefObject } from "react";
import { createPortal } from "react-dom";

interface Props {
  open: boolean;
  /** The element the popover hangs from (the trigger button's wrapper). */
  anchorRef: RefObject<HTMLElement | null>;
  onClose: () => void;
  align?: "left" | "right";
  minWidth?: number;
  children: ReactNode;
}

interface Pos {
  left: number;
  width: number;
  maxHeight: number;
  top?: number;
  bottom?: number;
  up: boolean;
}

/**
 * Dropdown panel rendered in a portal on <body> with fixed positioning.
 * Cards use `overflow-hidden`, which clips any normal absolutely-positioned dropdown —
 * a portal escapes that, flips upward when there's no room below, and scrolls internally
 * if the list is longer than the window.
 */
export function Popover({ open, anchorRef, onClose, align = "left", minWidth = 0, children }: Props) {
  const panelRef = useRef<HTMLDivElement>(null);
  const closeRef = useRef(onClose);
  closeRef.current = onClose;
  const [pos, setPos] = useState<Pos | null>(null);

  const place = useCallback(() => {
    const a = anchorRef.current;
    if (!a) return;
    const r = a.getBoundingClientRect();
    const M = 8;
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const width = Math.min(Math.max(r.width, minWidth), vw - M * 2);
    const left = Math.max(M, Math.min(align === "right" ? r.right - width : r.left, vw - width - M));
    const below = vh - r.bottom - M - 6;
    const above = r.top - M - 6;
    const up = below < 280 && above > below;
    const maxHeight = Math.max(160, Math.min(up ? above : below, 560));
    setPos(up ? { left, width, maxHeight, bottom: vh - r.top + 6, up } : { left, width, maxHeight, top: r.bottom + 6, up });
  }, [anchorRef, align, minWidth]);

  useLayoutEffect(() => {
    if (!open) return;
    place();
    window.addEventListener("resize", place);
    window.addEventListener("scroll", place, true);
    return () => {
      window.removeEventListener("resize", place);
      window.removeEventListener("scroll", place, true);
    };
  }, [open, place]);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      const t = e.target as Node;
      if (panelRef.current?.contains(t) || anchorRef.current?.contains(t)) return;
      closeRef.current();
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && closeRef.current();
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open, anchorRef]);

  return createPortal(
    <AnimatePresence>
      {open && pos && (
        <motion.div
          ref={panelRef}
          role="listbox"
          initial={{ opacity: 0, y: pos.up ? 6 : -6, scale: 0.98 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: pos.up ? 6 : -6, scale: 0.98 }}
          transition={{ duration: 0.15 }}
          style={{ position: "fixed", left: pos.left, top: pos.top, bottom: pos.bottom, width: pos.width, maxHeight: pos.maxHeight, transformOrigin: pos.up ? "bottom" : "top" }}
          className="z-[100] overflow-y-auto overscroll-contain rounded-xl border border-cyan-400/20 bg-[#070a12]/95 p-1.5 shadow-[0_20px_50px_-10px_rgba(0,0,0,0.9),0_0_30px_-10px_rgba(34,211,238,0.3)] backdrop-blur-xl"
        >
          {children}
        </motion.div>
      )}
    </AnimatePresence>,
    document.body,
  );
}
