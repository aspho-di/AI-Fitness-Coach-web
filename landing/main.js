/* ── Custom cursor ─────────────────────────────────────────────────────── */
const cursor     = document.getElementById('cursor');
const cursorRing = document.getElementById('cursor-ring');
let mx = 0, my = 0, rx = 0, ry = 0;

document.addEventListener('mousemove', e => {
  mx = e.clientX;
  my = e.clientY;
});

function animateCursor() {
  cursor.style.left = mx + 'px';
  cursor.style.top  = my + 'px';
  rx += (mx - rx) * 0.12;
  ry += (my - ry) * 0.12;
  cursorRing.style.left = rx + 'px';
  cursorRing.style.top  = ry + 'px';
  requestAnimationFrame(animateCursor);
}
animateCursor();


/* ── Scroll reveal ─────────────────────────────────────────────────────── */
const revealObserver = new IntersectionObserver((entries) => {
  entries.forEach((entry, i) => {
    if (entry.isIntersecting) {
      entry.target.style.transitionDelay = (i * 0.06) + 's';
      entry.target.classList.add('visible');
    }
  });
}, { threshold: 0.1 });

document.querySelectorAll('.reveal').forEach(el => revealObserver.observe(el));


/* ── Skeleton squat animation ──────────────────────────────────────────── */
let squats = 0;
let angle  = 160;
let going  = 'down';

const animCount = document.getElementById('anim-count');
const animAngle = document.getElementById('anim-angle');
const animBar   = document.getElementById('anim-bar');
const animStage = document.getElementById('anim-stage');

// Skeleton element refs
const sk = {
  head:   document.getElementById('sk-head'),
  neck:   document.getElementById('sk-neck'),
  sbar:   document.getElementById('sk-sbar'),
  torso:  document.getElementById('sk-torso'),
  hbar:   document.getElementById('sk-hbar'),
  larm1:  document.getElementById('sk-larm1'),
  larm2:  document.getElementById('sk-larm2'),
  rarm1:  document.getElementById('sk-rarm1'),
  rarm2:  document.getElementById('sk-rarm2'),
  lleg1:  document.getElementById('sk-lleg1'),
  lleg2:  document.getElementById('sk-lleg2'),
  rleg1:  document.getElementById('sk-rleg1'),
  rleg2:  document.getElementById('sk-rleg2'),
  chest:  document.getElementById('sk-chest'),
  hip:    document.getElementById('sk-hip'),
  lknee:  document.getElementById('sk-lknee'),
  rknee:  document.getElementById('sk-rknee'),
  lelbow: document.getElementById('sk-lelbow'),
  relbow: document.getElementById('sk-relbow'),
};

/**
 * Rebuild the skeleton using fixed bone lengths and a knee angle.
 * Ankles are pinned to the ground. Everything above is computed
 * upward via forward-kinematics so the body DESCENDS — no stretching.
 *
 * t = 0  →  standing upright  (knee angle ≈ 160°)
 * t = 1  →  deep squat        (knee angle ≈  60°)
 */
function updateSkeleton(t) {
  // ── Bone lengths (SVG units) ──────────────────────────────────────────
  const THIGH  = 32;   // hip   → knee
  const SHIN   = 30;   // knee  → ankle
  const TORSO  = 32;   // hip   → shoulder
  const HEAD_R = 7;

  // ── Ankle positions are FIXED (ground level, no spreading) ───────────
  const lAnkleX = 26;
  const rAnkleX = 54;
  const ankleY  = 120;

  // ── Hip X positions are also fixed (pelvis doesn't widen) ────────────
  const lHipX = 30;
  const rHipX = 50;
  const hipX  = 40;

  // ── Compute hip Y by solving two-bone IK (thigh + shin) ──────────────
  // Both ankle and hip X are fixed; we find hip Y such that
  // the chain ankle→knee→hip has bones of length SHIN and THIGH.
  // Distance ankle→hip (horizontal) is fixed:
  const dx = hipX - lAnkleX; // = 14 px
  const legLen = THIGH + SHIN; // max reach

  // Hip descends as t increases. At t=0 hip is near the top, at t=1 it sinks.
  // We compute hip Y from the squat depth directly:
  //   hipY ranges from (ankleY - legLen) standing to a lower value squatting.
  // Simple: hipY = ankleY - legLen + t * (legLen - SHIN * 0.85)
  const hipY = ankleY - legLen + t * (legLen - SHIN * 0.9);

  // ── Knee position via two-bone IK ─────────────────────────────────────
  // Given ankle (lAnkleX, ankleY) and hip (lHipX, hipY),
  // find knee so that |ankle→knee| = SHIN and |knee→hip| = THIGH.
  // Knees always protrude FORWARD (positive X offset from the line).
  function solveKnee(ankX, ankY, hX, hY) {
    const dx = hX - ankX;
    const dy = hY - ankY;
    const dist = Math.sqrt(dx * dx + dy * dy);
    // Clamp dist to valid range
    const d = Math.min(dist, THIGH + SHIN - 0.1);
    // Cosine rule: angle at ankle
    const cosA = (d * d + SHIN * SHIN - THIGH * THIGH) / (2 * d * SHIN);
    const angA  = Math.acos(Math.max(-1, Math.min(1, cosA)));
    // Direction from ankle to hip
    const baseAng = Math.atan2(dy, dx);
    // Knee is at ankle + SHIN rotated by +angA (forward = left side protrudes right, right side protrudes left)
    const kneeAng = baseAng - angA; // subtract → knee goes forward/outward
    return {
      x: ankX + Math.cos(kneeAng) * SHIN,
      y: ankY + Math.sin(kneeAng) * SHIN,
    };
  }

  const lKnee = solveKnee(lAnkleX, ankleY, lHipX, hipY);
  const rKnee = solveKnee(rAnkleX, ankleY, rHipX, hipY);
  const lKneeX = lKnee.x, lKneeY = lKnee.y;
  const rKneeX = rKnee.x, rKneeY = rKnee.y;

  // ── Torso lean (forward to counterbalance) ────────────────────────────
  const torsoLeanRad = t * 0.25; // max ~14° forward tilt
  const shoulderX = hipX + Math.sin(torsoLeanRad) * TORSO;
  const shoulderY = hipY - Math.cos(torsoLeanRad) * TORSO;

  // ── Head ──────────────────────────────────────────────────────────────
  const neckLen = 8;
  const headX = shoulderX + Math.sin(torsoLeanRad) * neckLen;
  const headY = shoulderY - Math.cos(torsoLeanRad) * neckLen - HEAD_R;

  // ── Arms: parallel to floor, both pointing forward ───────────────────
  // At rest (t=0): arms hang straight down (angle = 90° from horizontal)
  // At squat (t=1): arms fully horizontal forward (angle = 0°)
  const armAng = (1 - t) * (Math.PI / 2); // 90° → 0°

  const UPPER_ARM = 18;
  const FORE_ARM  = 16;

  const lShoulderX = shoulderX - 20;
  const rShoulderX = shoulderX + 20;

  // Both arms extend to the LEFT (forward in this side-view)
  const lElbowX = lShoulderX - Math.cos(armAng) * UPPER_ARM;
  const lElbowY = shoulderY  + Math.sin(armAng) * UPPER_ARM;
  const lWristX = lElbowX   - Math.cos(armAng) * FORE_ARM;
  const lWristY = lElbowY   + Math.sin(armAng) * FORE_ARM;

  const rElbowX = rShoulderX - Math.cos(armAng) * UPPER_ARM;
  const rElbowY = shoulderY  + Math.sin(armAng) * UPPER_ARM;
  const rWristX = rElbowX   - Math.cos(armAng) * FORE_ARM;
  const rWristY = rElbowY   + Math.sin(armAng) * FORE_ARM;

  // ── Apply to SVG ───────────────────────────────────────────────────────
  setAttr(sk.head,   { cx: headX,     cy: headY });
  setLine(sk.neck,   headX, headY + HEAD_R, shoulderX, shoulderY);
  setLine(sk.sbar,   shoulderX - 20, shoulderY, shoulderX + 20, shoulderY);
  setLine(sk.torso,  shoulderX, shoulderY, hipX, hipY);
  setLine(sk.hbar,   hipX - 14, hipY, hipX + 14, hipY);

  // Arms
  setLine(sk.larm1,  lShoulderX, shoulderY, lElbowX, lElbowY);
  setLine(sk.larm2,  lElbowX, lElbowY, lWristX, lWristY);
  setLine(sk.rarm1,  rShoulderX, shoulderY, rElbowX, rElbowY);
  setLine(sk.rarm2,  rElbowX, rElbowY, rWristX, rWristY);

  // Legs
  setLine(sk.lleg1,  lHipX, hipY, lKneeX, lKneeY);
  setLine(sk.lleg2,  lKneeX, lKneeY, lAnkleX, ankleY);
  setLine(sk.rleg1,  rHipX, hipY, rKneeX, rKneeY);
  setLine(sk.rleg2,  rKneeX, rKneeY, rAnkleX, ankleY);

  // Joints
  setAttr(sk.chest,  { cx: shoulderX, cy: shoulderY });
  setAttr(sk.hip,    { cx: hipX,      cy: hipY });
  setAttr(sk.lknee,  { cx: lKneeX,   cy: lKneeY });
  setAttr(sk.rknee,  { cx: rKneeX,   cy: rKneeY });
  setAttr(sk.lelbow, { cx: lElbowX,  cy: lElbowY });
  setAttr(sk.relbow, { cx: rElbowX,  cy: rElbowY });}

function setLine(el, x1, y1, x2, y2) {
  if (!el) return;
  el.setAttribute('x1', x1); el.setAttribute('y1', y1);
  el.setAttribute('x2', x2); el.setAttribute('y2', y2);
}
function setAttr(el, attrs) {
  if (!el) return;
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
}

setInterval(() => {
  going === 'down' ? angle -= 3 : angle += 3;

  if (angle <= 60)  going = 'up';
  if (angle >= 160) {
    going = 'down';
    squats++;
    animCount.textContent = squats;
  }

  animAngle.textContent = angle + '°';

  // Progress bar: 160° = 0%, 60° = 100%
  const pct = Math.round(((160 - angle) / (160 - 60)) * 100);
  animBar.style.width = pct + '%';

  if (angle < 110) {
    animStage.textContent = 'DOWN';
    animStage.style.color = 'var(--blue)';
  } else {
    animStage.textContent = 'UP';
    animStage.style.color = 'var(--accent)';
  }

  // Update skeleton: t = 0 (upright) → 1 (deep squat)
  const t = (160 - angle) / (160 - 60);
  updateSkeleton(Math.max(0, Math.min(1, t)));

}, 60);