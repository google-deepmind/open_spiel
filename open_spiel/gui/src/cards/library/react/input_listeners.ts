// Copyright 2026 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// The listeners on a scene root (input.ts says what they resolve and why):
// clicks and double clicks, the context menu and long press, pointer drags,
// hover, keys and focus, each turned into TableEvents.
import type {Point} from '../core/identity.ts';
import type {TableEvent} from '../core/events.ts';
import type {Target} from '../core/targets.ts';
import type {PointerInput} from '../dnd/controller.ts';
import {
  isActionable,
  MODIFIER_KEYS,
  modifiersOf,
  pointerVia,
  pointOf,
  resolveElement,
  resolvePointer,
  type Resolved,
  type Via,
} from './input.ts';
import type {SceneRuntimeImpl} from './scene_runtime.ts';

interface Press {
  x: number;
  y: number;
  pointerType: 'mouse' | 'touch' | 'pen';
  resolved: Resolved;
  fired: boolean;
  e: PointerEvent;
}

interface Hovered {
  tableId: string;
  key: string;
  target: Target;
}

const NO_MODIFIERS = {shift: false, ctrl: false, alt: false, meta: false};

const ARROWS: Readonly<Record<string, 'up' | 'down' | 'left' | 'right'>> = {
  ArrowUp: 'up',
  ArrowDown: 'down',
  ArrowLeft: 'left',
  ArrowRight: 'right',
};

class InputListeners {
  private clickDoubled = false;
  private longTimer: ReturnType<typeof setTimeout> | null = null;
  private press: Press | null = null;
  private suppressClick = false;
  /** A touch/pen long press fired: the native contextmenu for the same
   *  gesture is swallowed (one gesture, one event). */
  private swallowMenu = false;
  private hovered: Hovered | null = null;
  private unwatch: (() => void) | null = null;

  private readonly rt: SceneRuntimeImpl;
  private readonly root: HTMLElement;

  constructor(rt: SceneRuntimeImpl, root: HTMLElement) {
    this.rt = rt;
    this.root = root;
  }

  private base(r: Resolved, via: Via, e: Event, mouse?: MouseEvent) {
    const at = mouse
      ? pointOf(this.rt, r.tableId, mouse)
      : {point: null, client: null};
    return {
      target: r.target,
      via,
      modifiers: mouse ? modifiersOf(mouse) : modifiersOf(e as KeyboardEvent),
      point: at.point,
      client: at.client,
      native: e,
      tableId: r.tableId,
    };
  }

  readonly clearLong = (): void => {
    if (this.longTimer !== null) clearTimeout(this.longTimer);
    this.longTimer = null;
    this.unwatch?.();
  };

  // Whether the pointer has moved past the drag threshold of the press.
  private pastThreshold(e: PointerEvent): boolean {
    if (!this.press) return false;
    const st = this.rt.style().interaction;
    const limit =
      this.press.pointerType === 'touch'
        ? st.touchDragThresholdPx
        : st.dragThresholdPx;
    return (
      Math.hypot(e.clientX - this.press.x, e.clientY - this.press.y) > limit
    );
  }

  /** While the long-press timer runs, watch the whole window: the pointer
   *  may leave the root before it passes the drag threshold or is released,
   *  and neither may then turn into a long press. */
  private watchPress(): void {
    this.unwatch?.();
    const mv = (e: PointerEvent): void => {
      if (this.longTimer === null) return;
      if (this.pastThreshold(e)) this.clearLong();
    };
    const end = (): void => {
      this.clearLong();
    };
    window.addEventListener('pointermove', mv, true);
    window.addEventListener('pointerup', end, true);
    window.addEventListener('pointercancel', end, true);
    this.unwatch = () => {
      window.removeEventListener('pointermove', mv, true);
      window.removeEventListener('pointerup', end, true);
      window.removeEventListener('pointercancel', end, true);
      this.unwatch = null;
    };
  }

  private input(e: PointerEvent | KeyboardEvent, client: Point): PointerInput {
    return {
      client,
      modifiers: modifiersOf(e),
      pointerType: e instanceof KeyboardEvent ? 'mouse' : pointerVia(e),
      native: e,
    };
  }

  readonly onClick = (e: MouseEvent): void => {
    if (this.suppressClick) {
      this.suppressClick = false;
      return;
    }
    if (e.button !== 0) return;
    const r = resolvePointer(this.rt, e);
    if (!r?.policy || !isActionable(r)) return;
    if (e.detail >= 3) return;
    if (e.detail === 2) {
      this.clickDoubled = true;
      this.rt.emit({
        ...this.base(r, pointerVia(e), e, e),
        type: 'doubleActivate',
      });
      return;
    }
    this.clickDoubled = false;
    this.rt.emit({...this.base(r, pointerVia(e), e, e), type: 'activate'});
  };

  readonly onDblClick = (e: MouseEvent): void => {
    if (this.clickDoubled) {
      this.clickDoubled = false;
      return;
    }
    const r = resolvePointer(this.rt, e);
    if (!r?.policy || !isActionable(r)) return;
    this.rt.emit({...this.base(r, 'mouse', e, e), type: 'doubleActivate'});
  };

  /** Fire the pending long press now (from the timer, or from the native
   *  contextmenu a touch browser sends for the same hold). */
  private fireLongPress(native: Event): void {
    const press = this.press;
    if (!press || press.fired) return;
    this.clearLong();
    // A drag that started suppresses the long press.
    if (this.rt.drag.state().source !== null) return;
    press.fired = true;
    // A long press fires first and suppresses the drag; on touch and pen
    // it also stands in for the tap (the click that follows is dropped).
    this.rt.drag.cancel('pointercancel');
    if (press.pointerType !== 'mouse') {
      this.suppressClick = true;
      this.swallowMenu = true;
    }
    this.rt.emit({
      ...this.base(press.resolved, press.pointerType, native, press.e),
      type: 'longPress',
    });
  }

  readonly onContextMenu = (e: MouseEvent): void => {
    const via = pointerVia(e);
    const press = this.press;
    const touchLike =
      via !== 'mouse' || (press !== null && press.pointerType !== 'mouse');
    if (touchLike && press && !press.fired) {
      // The browser's own long-press menu for a hold we are timing: it IS the long press.
      e.preventDefault();
      this.fireLongPress(e);
      return;
    }
    if (touchLike && this.swallowMenu) {
      this.swallowMenu = false;
      e.preventDefault();
      return;
    }
    const r = resolvePointer(this.rt, e, true);
    if (!r?.policy) return;
    if (isActionable(r)) e.preventDefault();
    this.rt.emit({...this.base(r, via, e, e), type: 'contextMenu'});
  };

  readonly onPointerDown = (e: PointerEvent): void => {
    // A new gesture: whatever the last one left to swallow belonged to it
    // (its click or its menu, had the browser sent one, came before this).
    this.suppressClick = false;
    this.swallowMenu = false;
    if (e.button !== 0) return;
    this.clearLong();
    this.press = null;
    const act = resolvePointer(this.rt, e);
    const acting = act?.policy !== undefined && isActionable(act);
    // An observed (non-interactive) card still long-presses; it never drags.
    const r = acting ? act : resolvePointer(this.rt, e, true);
    if (!r?.policy || !isActionable(r)) return;
    this.press = {
      x: e.clientX,
      y: e.clientY,
      pointerType: pointerVia(e),
      resolved: r,
      fired: false,
      e,
    };
    const st = this.rt.style().interaction;
    this.longTimer = setTimeout(() => {
      this.longTimer = null;
      this.fireLongPress(e);
    }, st.longPressMs);
    this.watchPress();
    if (acting && r.target.kind === 'card' && r.policy.canDrag) {
      this.rt.drag.press(
        r.target.ref,
        r.tableId,
        this.input(e, {x: e.clientX, y: e.clientY}),
      );
    }
  };

  readonly onPointerMove = (e: PointerEvent): void => {
    if (this.press && this.longTimer !== null && this.pastThreshold(e)) {
      this.clearLong();
    }
    if (this.press) {
      this.rt.drag.move(this.input(e, {x: e.clientX, y: e.clientY}));
    }
  };

  readonly onPointerUp = (e: PointerEvent): void => {
    this.clearLong();
    if (this.press) {
      this.rt.drag.release(this.input(e, {x: e.clientX, y: e.clientY}));
    }
    this.press = null;
  };

  readonly onPointerCancel = (): void => {
    this.clearLong();
    if (this.press) this.rt.drag.cancel('pointercancel');
    this.press = null;
    this.suppressClick = false;
  };

  private hoverOf(e: PointerEvent): Hovered | null {
    const r = resolveElement(this.rt, e.target as Element | null, true);
    if (!r?.policy) return null;
    const t = r.target;
    let key = '';
    if (t.kind === 'card') key = `card:${t.ref.id}`;
    else if (t.kind === 'slot') key = `slot:${t.slot}`;
    else if (t.kind === 'region') key = `region:${t.region}`;
    return key ? {tableId: r.tableId, key, target: t} : null;
  }

  private resolvedOf(h: Hovered): Resolved {
    return {
      tableId: h.tableId,
      target: h.target,
      scene: this.rt.table(h.tableId).scene,
      policy: this.rt.policy(h.tableId),
    };
  }

  private setHover(next: Hovered | null, e: PointerEvent): void {
    const hovered = this.hovered;
    if ((hovered?.key ?? null) === (next?.key ?? null)) return;
    if (hovered && this.rt.hasTable(hovered.tableId)) {
      this.rt.emit({
        ...this.base(this.resolvedOf(hovered), pointerVia(e), e, e),
        type: 'hoverEnd',
      });
    }
    this.hovered = next;
    const tags = new Map<string, readonly string[]>();
    if (next) {
      // The hover look marks what a click would act on: an observed
      // (non-interactive) card gets the events but not the look.
      const t = next.target;
      // A game may keep an interactive card unlit (policy.hoverLook).
      const r = this.resolvedOf(next);
      const actsOn =
        t.kind !== 'card' ||
        (!!r.scene.byId(t.ref.id)?.interactive &&
          (r.policy?.hoverLook?.(t.ref) ?? true));
      if (actsOn) tags.set(t.kind === 'card' ? t.ref.id : next.key, ['hover']);
      this.rt.emit({...this.base(r, pointerVia(e), e, e), type: 'hoverStart'});
    }
    this.rt.setLibraryTags('hover', tags);
  }

  readonly onPointerOver = (e: PointerEvent): void => {
    this.setHover(this.hoverOf(e), e);
  };

  readonly onPointerOut = (e: PointerEvent): void => {
    const to = e.relatedTarget as Element | null;
    if (to && this.root.contains(to)) return; // pointerover on the new element handles it
    this.setHover(null, e);
  };

  private keyTarget(e: Event): Resolved | null {
    const el = e.target as Element | null;
    if (!el) return null;
    // Keys inside region content (an input in an action bar) are the game's.
    const node = el.closest('[data-card-id],[data-slot-id],[data-region-id]');
    if (!node || node.hasAttribute('data-region-id')) return null;
    const r = resolveElement(this.rt, el);
    return r ?? null;
  }

  // A keyboard drag owns the arrows, Tab, Enter, Space and Esc. Whether the
  // key was the drag's.
  private dragKey(e: KeyboardEvent): boolean {
    if (e.key === 'Escape') {
      e.preventDefault();
      this.rt.drag.cancel('escape');
      return true;
    }
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      this.rt.drag.keyDrop();
      return true;
    }
    const dir = ARROWS[e.key];
    if (dir) {
      e.preventDefault();
      this.rt.drag.keyCycle(dir);
      return true;
    }
    if (e.key === 'Tab') {
      e.preventDefault();
      this.rt.drag.keyCycle(e.shiftKey ? 'prev' : 'next');
      return true;
    }
    return false;
  }

  readonly onKeyDown = (e: KeyboardEvent): void => {
    if (e.defaultPrevented) return;
    const r = this.keyTarget(e);
    if (!r?.policy) return;
    const kb = this.base(r, 'keyboard', e);
    // Otherwise the focus model takes the navigation keys first.
    const dragging = this.rt.drag.state().source !== null;
    if (!dragging && this.rt.focus.onKey(e, r.scene, r.policy) === 'handled') {
      return;
    }
    if (dragging && this.dragKey(e)) return;
    if (e.key === 'Enter') {
      e.preventDefault();
      this.rt.emit({...kb, type: e.shiftKey ? 'doubleActivate' : 'activate'});
      return;
    }
    if (e.key === ' ' || e.key === 'Spacebar') {
      e.preventDefault();
      const t = r.target;
      if (
        t.kind === 'card' &&
        r.policy.canDrag &&
        this.rt.drag.keyPick(t.ref, r.tableId)
      ) {
        return;
      }
      this.rt.emit({...kb, type: 'activate'});
      return;
    }
    if ((e.key === 'F10' && e.shiftKey) || e.key === 'ContextMenu') {
      e.preventDefault();
      this.rt.emit({...kb, type: 'contextMenu'});
      return;
    }
    if (MODIFIER_KEYS.has(e.key)) return;
    this.rt.emit({...kb, type: 'key', key: e.key});
  };

  readonly onFocusIn = (e: FocusEvent): void => {
    const r = this.keyTarget(e);
    if (!r?.policy) return;
    // The focus model owns the roving tab stop and the (focus-visible) ring.
    this.rt.focus.set(r.target);
    const event: TableEvent = {
      ...this.base(r, 'keyboard', e),
      type: 'focus',
      modifiers: NO_MODIFIERS,
    };
    this.rt.emit(event);
  };

  readonly onFocusOut = (e: FocusEvent): void => {
    const r = this.keyTarget(e);
    const to = e.relatedTarget as Element | null;
    if (!to || !this.root.contains(to)) {
      this.rt.setLibraryTags('focus', new Map());
    }
    if (!r?.policy) return;
    const event: TableEvent = {
      ...this.base(r, 'keyboard', e),
      type: 'blur',
      modifiers: NO_MODIFIERS,
    };
    this.rt.emit(event);
  };

  readonly onWindowBlur = (): void => {
    this.clearLong();
    if (this.rt.drag.state().source !== null) this.rt.drag.cancel('blur');
    this.press = null;
  };
}

/** Attach the listeners to the scene root. Returns the detach function. */
export function attachInput(
  rt: SceneRuntimeImpl,
  root: HTMLElement,
): () => void {
  const l = new InputListeners(rt, root);
  const listeners: Array<[string, (e: never) => void]> = [
    ['click', l.onClick],
    ['dblclick', l.onDblClick],
    ['contextmenu', l.onContextMenu],
    ['pointerdown', l.onPointerDown],
    ['pointermove', l.onPointerMove],
    ['pointerup', l.onPointerUp],
    ['pointercancel', l.onPointerCancel],
    ['pointerover', l.onPointerOver],
    ['pointerout', l.onPointerOut],
    ['keydown', l.onKeyDown],
    ['focusin', l.onFocusIn],
    ['focusout', l.onFocusOut],
  ];
  for (const [type, f] of listeners) {
    root.addEventListener(type, f as EventListener);
  }
  window.addEventListener('blur', l.onWindowBlur);
  return () => {
    l.clearLong();
    for (const [type, f] of listeners) {
      root.removeEventListener(type, f as EventListener);
    }
    window.removeEventListener('blur', l.onWindowBlur);
  };
}
