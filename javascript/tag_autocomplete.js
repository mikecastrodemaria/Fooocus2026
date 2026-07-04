// custom-12 — Tag Autocomplete (tags booru + assets locaux)
// Injecte par modules/ui_gradio_extensions.py uniquement si
// tag_autocomplete.enabled = true dans config.txt.
//
// Sources de donnees (servies via /file=, voir modules/tag_autocomplete.py) :
//   - tags/danbooru.csv, tags/e621.csv : nom,categorie,count,"alias1,alias2"
//   - tags/local_assets.json : { loras, embeddings, wildcards }
//
// UX : dropdown sous le caret dans #positive_prompt / #negative_prompt.
//   Fleches = naviguer, Tab/Entree = inserer, Echap = fermer, `__` = wildcards.

(function () {
    'use strict';

    const meta = document.querySelector('meta[name="ta-config"]');
    if (!meta) return;

    let CFG;
    try { CFG = JSON.parse(meta.getAttribute('content')); } catch (e) {
        console.warn('[TagAC] ta-config illisible', e);
        return;
    }
    const MIN_CHARS = Math.max(1, CFG.minChars || 2);
    const MAX_RESULTS = Math.max(1, CFG.maxResults || 12);

    // ------------------------------------------------------------------ data
    // booru: tableaux paralleles pour rester sobre en memoire (~200k tags).
    const T_NAME = [], T_CAT = [], T_COUNT = [], T_ALIAS = [];
    // locals: petites listes, objets simples.
    let LORAS = [], EMBEDDINGS = [], WILDCARDS = [];
    let dataReady = false;

    function parseCsvLine(line) {
        // Format: nom,cat,count,"a,b,c" (dernier champ optionnel, guillemets optionnels)
        const parts = [];
        let cur = '', inQ = false;
        for (let i = 0; i < line.length; i++) {
            const ch = line[i];
            if (inQ) { if (ch === '"') inQ = false; else cur += ch; }
            else if (ch === '"') inQ = true;
            else if (ch === ',') { parts.push(cur); cur = ''; }
            else cur += ch;
        }
        parts.push(cur);
        return parts;
    }

    async function loadCsv(url) {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`${url} -> HTTP ${res.status}`);
        const text = await res.text();
        let n = 0;
        for (const line of text.split('\n')) {
            if (!line || line.length < 3) continue;
            const p = parseCsvLine(line.trim());
            if (p.length < 3) continue;
            T_NAME.push(p[0]);
            T_CAT.push(parseInt(p[1], 10) || 0);
            T_COUNT.push(parseInt(p[2], 10) || 0);
            T_ALIAS.push(p[3] ? p[3].split(',') : null);
            n++;
        }
        return n;
    }

    async function loadAll() {
        const jobs = [];
        for (const [name, url] of Object.entries(CFG.sources || {})) {
            jobs.push(loadCsv(url)
                .then(n => console.log(`[TagAC] ${name}: ${n} tags`))
                .catch(e => console.warn(`[TagAC] source ${name} indisponible`, e)));
        }
        if (CFG.localAssets) {
            jobs.push(fetch(CFG.localAssets).then(r => r.json()).then(d => {
                LORAS = d.loras || [];
                EMBEDDINGS = d.embeddings || [];
                WILDCARDS = d.wildcards || [];
                console.log(`[TagAC] locaux: ${LORAS.length} loras, ${EMBEDDINGS.length} embeddings, ${WILDCARDS.length} wildcards`);
            }).catch(e => console.warn('[TagAC] local_assets.json indisponible', e)));
        }
        await Promise.all(jobs);
        dataReady = true;
    }

    // ---------------------------------------------------------------- search
    // Resultat: {label, insert, kind, cat, count}
    // kind: 'tag' | 'alias' | 'lora' | 'emb' | 'wc'

    function tagToInsertion(name) {
        let s = name;
        if (CFG.replaceUnderscores) s = s.replace(/_/g, ' ');
        return s.replace(/\(/g, '\\(').replace(/\)/g, '\\)');
    }

    function searchBooru(q, prefixOut, subOut, budget) {
        for (let i = 0; i < T_NAME.length; i++) {
            const name = T_NAME[i];
            let hit = null;
            if (name.startsWith(q)) hit = 'p';
            else if (name.includes(q)) hit = 's';
            else if (T_ALIAS[i]) {
                for (const a of T_ALIAS[i]) {
                    if (a.startsWith(q)) {
                        prefixOut.push({ label: `${a} → ${name}`, insert: tagToInsertion(name), kind: 'alias', cat: T_CAT[i], count: T_COUNT[i] });
                        hit = 'done';
                        break;
                    }
                }
            }
            if (hit === 'p') prefixOut.push({ label: name, insert: tagToInsertion(name), kind: 'tag', cat: T_CAT[i], count: T_COUNT[i] });
            else if (hit === 's' && subOut.length < budget) subOut.push({ label: name, insert: tagToInsertion(name), kind: 'tag', cat: T_CAT[i], count: T_COUNT[i] });
        }
    }

    function searchLocals(q, out) {
        for (const lo of LORAS) {
            for (const t of lo.triggers) {
                // triggers inseres tels quels: les score_9 & co ne doivent pas etre alteres
                if (t.toLowerCase().replace(/ /g, '_').startsWith(q)) {
                    out.push({ label: `${t}  [${lo.name}]`, insert: t, kind: 'lora', cat: -1, count: 0 });
                }
            }
        }
        for (const em of EMBEDDINGS) {
            const stem = em.name.toLowerCase().replace(/ /g, '_');
            if (stem.startsWith(q) || em.triggers.some(t => t.toLowerCase().replace(/ /g, '_').startsWith(q))) {
                out.push({ label: `${em.name}  [embedding]`, insert: `(embedding:${em.name}:1.0)`, kind: 'emb', cat: -2, count: 0 });
            }
        }
    }

    function searchWildcards(q, out) {
        // q arrive sans les __ de tete
        for (const w of WILDCARDS) {
            if (w.toLowerCase().startsWith(q)) {
                out.push({ label: `__${w}__  [wildcard]`, insert: `__${w}__`, kind: 'wc', cat: -3, count: 0 });
            }
        }
    }

    function search(raw) {
        const results = [];
        if (raw.startsWith('__')) {
            searchWildcards(raw.slice(2).toLowerCase(), results);
            return results.slice(0, MAX_RESULTS);
        }
        const q = raw.toLowerCase().replace(/ /g, '_');
        if (q.length < MIN_CHARS) return results;
        searchLocals(q, results);                       // locaux d'abord: c'est TA bibliotheque
        const prefix = [], sub = [];
        searchBooru(q, prefix, sub, 200);
        prefix.sort((a, b) => b.count - a.count);
        sub.sort((a, b) => b.count - a.count);
        for (const r of prefix) { if (results.length >= MAX_RESULTS) break; results.push(r); }
        for (const r of sub) { if (results.length >= MAX_RESULTS) break; results.push(r); }
        return results;
    }

    // ------------------------------------------------------------------- ui
    const CAT_COLORS = {
        0: 'var(--ta-general, #6b9eff)',   // general
        1: 'var(--ta-artist, #e06c6c)',    // artiste
        3: 'var(--ta-copy, #b98cf0)',      // copyright
        4: 'var(--ta-char, #6cc76c)',      // personnage
        5: 'var(--ta-meta, #e0a96c)',      // meta / species e621
        '-1': '#f0c674',                   // lora trigger
        '-2': '#8fd3d3',                   // embedding
        '-3': '#d3a5e8',                   // wildcard
    };

    const style = document.createElement('style');
    style.textContent = `
    #tag-ac-popup {
        position: fixed; z-index: 10000; display: none;
        min-width: 260px; max-width: 460px; max-height: 340px; overflow-y: auto;
        background: var(--background-fill-primary, #1f1f28);
        border: 1px solid var(--border-color-primary, #444);
        border-radius: 8px; box-shadow: 0 6px 24px rgba(0,0,0,.45);
        font-family: var(--font, sans-serif); font-size: 13px;
    }
    #tag-ac-popup .ta-item {
        display: flex; justify-content: space-between; gap: 12px;
        padding: 4px 10px; cursor: pointer; white-space: nowrap;
    }
    #tag-ac-popup .ta-item .ta-name { overflow: hidden; text-overflow: ellipsis; }
    #tag-ac-popup .ta-item .ta-count { opacity: .55; font-size: 11px; }
    #tag-ac-popup .ta-item.ta-sel,
    #tag-ac-popup .ta-item:hover { background: var(--background-fill-secondary, #33333f); }
    `;
    document.head.appendChild(style);

    const popup = document.createElement('div');
    popup.id = 'tag-ac-popup';
    document.body.appendChild(popup);

    let state = { ta: null, items: [], sel: 0, tokenStart: 0 };

    function fmtCount(n) {
        if (!n) return '';
        if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
        if (n >= 1e3) return Math.round(n / 1e3) + 'k';
        return String(n);
    }

    function hide() {
        popup.style.display = 'none';
        state.items = [];
    }

    function render() {
        popup.innerHTML = '';
        state.items.forEach((it, i) => {
            const div = document.createElement('div');
            div.className = 'ta-item' + (i === state.sel ? ' ta-sel' : '');
            const name = document.createElement('span');
            name.className = 'ta-name';
            name.textContent = it.label;
            name.style.color = CAT_COLORS[it.cat] || 'inherit';
            const count = document.createElement('span');
            count.className = 'ta-count';
            count.textContent = fmtCount(it.count);
            div.appendChild(name); div.appendChild(count);
            div.addEventListener('mousedown', ev => { ev.preventDefault(); apply(i); });
            popup.appendChild(div);
        });
    }

    // Position du caret via un div-miroir (technique standard textarea).
    function caretViewportXY(ta) {
        const cs = getComputedStyle(ta);
        const mirror = document.createElement('div');
        for (const p of ['fontFamily', 'fontSize', 'fontWeight', 'letterSpacing',
            'lineHeight', 'textTransform', 'wordSpacing', 'paddingLeft', 'paddingTop',
            'paddingRight', 'paddingBottom', 'borderLeftWidth', 'borderTopWidth', 'boxSizing']) {
            mirror.style[p] = cs[p];
        }
        mirror.style.position = 'absolute';
        mirror.style.visibility = 'hidden';
        mirror.style.whiteSpace = 'pre-wrap';
        mirror.style.wordWrap = 'break-word';
        mirror.style.width = ta.clientWidth + 'px';
        mirror.textContent = ta.value.substring(0, ta.selectionEnd);
        const marker = document.createElement('span');
        marker.textContent = '​';
        mirror.appendChild(marker);
        document.body.appendChild(mirror);
        const rect = ta.getBoundingClientRect();
        const lineH = parseFloat(cs.lineHeight) || parseFloat(cs.fontSize) * 1.3;
        const x = rect.left + marker.offsetLeft - ta.scrollLeft;
        const y = rect.top + marker.offsetTop - ta.scrollTop + lineH;
        document.body.removeChild(mirror);
        return { x, y };
    }

    function show(ta) {
        render();
        popup.style.display = 'block';
        let { x, y } = caretViewportXY(ta);
        const pw = popup.offsetWidth, ph = popup.offsetHeight;
        if (x + pw > window.innerWidth - 8) x = window.innerWidth - pw - 8;
        if (y + ph > window.innerHeight - 8) y = Math.max(8, y - ph - 24);
        popup.style.left = Math.max(8, x) + 'px';
        popup.style.top = y + 'px';
    }

    // ------------------------------------------------------------- insertion
    const DELIM = /[,\n(){}\[\]|:>]/;

    function currentToken(ta) {
        const text = ta.value, caret = ta.selectionEnd;
        if (ta.selectionStart !== caret) return null; // selection active: on ne joue pas
        let s = caret;
        while (s > 0 && !DELIM.test(text[s - 1])) s--;
        while (s < caret && text[s] === ' ') s++;
        return { start: s, text: text.slice(s, caret) };
    }

    function apply(index) {
        const it = state.items[index];
        const ta = state.ta;
        if (!it || !ta) { hide(); return; }
        const caret = ta.selectionEnd;
        let insert = it.insert;
        if (CFG.insertComma) insert += ', ';
        ta.value = ta.value.slice(0, state.tokenStart) + insert + ta.value.slice(caret);
        const pos = state.tokenStart + insert.length;
        ta.selectionStart = ta.selectionEnd = pos;
        // indispensable pour que Gradio voie la nouvelle valeur
        ta.dispatchEvent(new Event('input', { bubbles: true }));
        hide();
        ta.focus();
    }

    // --------------------------------------------------------------- events
    let debounceTimer = null;

    function onInput(ev) {
        const ta = ev.target;
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
            if (!dataReady) return;
            const tok = currentToken(ta);
            if (!tok || (!tok.text.startsWith('__') && tok.text.length < MIN_CHARS)) { hide(); return; }
            const items = search(tok.text);
            if (!items.length) { hide(); return; }
            state = { ta, items, sel: 0, tokenStart: tok.start };
            show(ta);
        }, 80);
    }

    function onKeyDown(ev) {
        if (popup.style.display !== 'block' || ev.target !== state.ta) return;
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

    function attach(ta) {
        if (ta.dataset.taAc) return;
        ta.dataset.taAc = '1';
        ta.addEventListener('input', onInput);
        ta.addEventListener('keydown', onKeyDown, true); // capture: passer avant Gradio
        ta.addEventListener('blur', () => setTimeout(hide, 150));
        ta.addEventListener('click', hide);
    }

    function boot() {
        const found = [];
        for (const sel of ['#positive_prompt textarea', '#negative_prompt textarea']) {
            const ta = document.querySelector(sel);
            if (ta) { attach(ta); found.push(sel); }
        }
        return found.length === 2;
    }

    loadAll();
    const bootTimer = setInterval(() => { if (boot()) clearInterval(bootTimer); }, 500);
    window.addEventListener('resize', hide);
    document.addEventListener('scroll', hide, true);
})();
