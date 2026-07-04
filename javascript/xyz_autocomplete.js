// custom-15.1 — Autosuggest contextuel des champs de valeurs de la grille XYZ.
// Injecte par ui_gradio_extensions.py si job_queue.enabled. Les listes viennent
// du <meta name="xyz-ac"> : samplers/schedulers de flags, checkpoints et presets
// installes, valeurs de calibrage classiques pour les axes numeriques.
// La suggestion depend de l'axe choisi dans le dropdown voisin (#xyz_param_*).

(function () {
    'use strict';

    const meta = document.querySelector('meta[name="xyz-ac"]');
    if (!meta) return;
    let SUGG;
    try { SUGG = JSON.parse(meta.getAttribute('content')); } catch (e) {
        console.warn('[XYZ-AC] meta illisible', e);
        return;
    }

    const style = document.createElement('style');
    style.textContent = `
    #xyz-ac-popup {
        position: fixed; z-index: 10000; display: none;
        min-width: 220px; max-width: 460px; max-height: 300px; overflow-y: auto;
        background: var(--background-fill-primary, #1f1f28);
        border: 1px solid var(--border-color-primary, #444);
        border-radius: 8px; box-shadow: 0 6px 24px rgba(0,0,0,.45);
        font-family: var(--font, sans-serif); font-size: 13px;
    }
    #xyz-ac-popup .xa-item {
        padding: 4px 10px; cursor: pointer;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    #xyz-ac-popup .xa-item.xa-sel,
    #xyz-ac-popup .xa-item:hover { background: var(--background-fill-secondary, #33333f); }
    `;
    const popup = document.createElement('div');
    popup.id = 'xyz-ac-popup';

    let state = { el: null, items: [], sel: 0, tokenStart: 0 };

    function axisFor(which) {
        const inp = document.querySelector(`#xyz_param_${which} input`);
        return inp ? inp.value.trim() : '';
    }

    function hide() {
        popup.style.display = 'none';
        state.items = [];
    }

    function render() {
        popup.innerHTML = '';
        state.items.forEach((it, i) => {
            const div = document.createElement('div');
            div.className = 'xa-item' + (i === state.sel ? ' xa-sel' : '');
            div.textContent = it;
            div.addEventListener('mousedown', ev => { ev.preventDefault(); apply(i); });
            popup.appendChild(div);
        });
    }

    // token courant = ce qui suit la derniere virgule avant le caret
    function currentToken(el) {
        const caret = (el.selectionEnd === null || el.selectionEnd === undefined)
            ? el.value.length : el.selectionEnd;
        const upto = el.value.slice(0, caret);
        let s = upto.lastIndexOf(',') + 1;
        while (s < caret && upto[s] === ' ') s++;
        return { start: s, text: upto.slice(s), caret };
    }

    function apply(i) {
        const it = state.items[i], el = state.el;
        if (!it || !el) { hide(); return; }
        const caret = (el.selectionEnd === null || el.selectionEnd === undefined)
            ? el.value.length : el.selectionEnd;
        el.value = el.value.slice(0, state.tokenStart) + it + ', ' + el.value.slice(caret);
        const pos = state.tokenStart + it.length + 2;
        el.selectionStart = el.selectionEnd = pos;
        el.dispatchEvent(new Event('input', { bubbles: true }));
        hide();
        el.focus();
    }

    function suggest(el, which) {
        const list = SUGG[axisFor(which)];
        if (!list || !list.length) { hide(); return; }
        const tok = currentToken(el);
        const q = tok.text.toLowerCase();
        const used = el.value.toLowerCase().split(',').map(s => s.trim()).filter(Boolean);
        let items = list.filter(v =>
            v.toLowerCase().includes(q) && !used.includes(v.toLowerCase()));
        if (!items.length) { hide(); return; }
        state = { el, items: items.slice(0, 12), sel: 0, tokenStart: tok.start };
        render();
        popup.style.display = 'block';
        const r = el.getBoundingClientRect();
        let x = r.left, y = r.bottom + 4;
        const pw = popup.offsetWidth, ph = popup.offsetHeight;
        if (y + ph > window.innerHeight - 8) y = Math.max(8, r.top - ph - 4);
        if (x + pw > window.innerWidth - 8) x = window.innerWidth - pw - 8;
        popup.style.left = x + 'px';
        popup.style.top = y + 'px';
    }

    function onKeyDown(ev) {
        if (popup.style.display !== 'block' || ev.target !== state.el) return;
        switch (ev.key) {
            case 'ArrowDown':
                state.sel = (state.sel + 1) % state.items.length;
                render(); ev.preventDefault(); ev.stopPropagation(); break;
            case 'ArrowUp':
                state.sel = (state.sel - 1 + state.items.length) % state.items.length;
                render(); ev.preventDefault(); ev.stopPropagation(); break;
            case 'Tab':
            case 'Enter':
                apply(state.sel); ev.preventDefault(); ev.stopPropagation(); break;
            case 'Escape':
                hide(); ev.preventDefault(); ev.stopPropagation(); break;
        }
    }

    const attached = new Set();

    function attach(which) {
        if (attached.has(which)) return;
        const el = document.querySelector(`#xyz_vals_${which} input, #xyz_vals_${which} textarea`);
        if (!el) return;
        attached.add(which);
        el.setAttribute('autocomplete', 'off');
        el.spellcheck = false;
        el.addEventListener('focus', () => suggest(el, which));
        el.addEventListener('click', () => suggest(el, which));
        el.addEventListener('input', () => suggest(el, which));
        el.addEventListener('keydown', onKeyDown, true);
        el.addEventListener('blur', () => setTimeout(hide, 150));
    }

    function start() {
        (document.head || document.documentElement).appendChild(style);
        document.body.appendChild(popup);
        const timer = setInterval(() => {
            ['x', 'y', 'z'].forEach(attach);
            if (attached.size === 3) clearInterval(timer);
        }, 700);
        window.addEventListener('resize', hide);
        document.addEventListener('scroll', hide, true);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', start);
    } else {
        start();
    }
})();
