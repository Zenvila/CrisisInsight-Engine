/**
 * animations.js — Micro-animations & Counter Effects
 */

// ── Counter Animation ──
function animateCounters(container) {
    const counters = (container || document).querySelectorAll('.counter');
    counters.forEach(el => {
        const target = parseFloat(el.dataset.target);
        if (isNaN(target)) return;
        const duration = 1200;
        const start = performance.now();
        const isDecimal = target % 1 !== 0;

        function step(now) {
            const elapsed = now - start;
            const progress = Math.min(elapsed / duration, 1);
            const ease = 1 - Math.pow(1 - progress, 3); // ease-out cubic
            const current = target * ease;
            el.textContent = isDecimal ? current.toFixed(1) : Math.round(current);
            if (progress < 1) requestAnimationFrame(step);
        }
        requestAnimationFrame(step);
    });
}

// ── Intersection Observer for fade-in ──
const fadeObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, { threshold: 0.1 });

document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.glass-card').forEach(card => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        card.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
        fadeObserver.observe(card);
    });
});
