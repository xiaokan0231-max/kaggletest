const neurogolfHtml = `
<div class="topics-header-bar">
    <h2>⛳ NeuroGolf 400 任务追踪大盘</h2>
    <button onclick="refreshNeuroGolf(this)" style="padding:0.35rem 0.8rem;border:1px solid var(--border-color);border-radius:6px;background:var(--bg-card);color:var(--text-primary);cursor:pointer;font-size:0.85rem;">🔄 刷新</button>
</div>
<div id="neuro-family-bar" style="display:flex;flex-wrap:wrap;gap:0.4rem;margin-bottom:1rem;"></div>
<p style="color:var(--text-muted);font-size:0.85rem;margin-bottom:1.5rem;">状态以工作区模型与认领台账为准（已完成 = 非 dummy ONNX 在库；进行中 = 有效认领），论坛帖仅作跳转链接。</p>

<div class="kpi-cards" id="neuro-kpis" style="margin-bottom: 2rem;"></div>

<div id="neuro-kaggle-section" style="margin-bottom:2rem;padding:1rem;border:1px solid var(--border-color);border-radius:8px;background:var(--bg-card);">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
        <div style="display:flex;align-items:center;gap:0.5rem;">
            <strong>🏆 Kaggle 提交</strong>
            <button onclick="refreshKaggleSection(this)" title="刷新榜单" style="padding:0.2rem 0.5rem;border:1px solid var(--border-color);border-radius:5px;background:var(--bg-card-hover);color:var(--text-muted);cursor:pointer;font-size:0.8rem;">🔄</button>
        </div>
        <div style="display:flex;gap:0.5rem;align-items:center;flex-wrap:wrap;">
            <input type="text" id="kaggle-message" value="Hub auto submit"
                   style="padding:0.3rem 0.6rem;border:1px solid var(--border-color);border-radius:6px;background:var(--bg-input,var(--bg-card-hover));color:var(--text-primary);font-size:0.85rem;width:200px;">
            <button id="kaggle-submit-btn" onclick="submitToKaggle()"
                    style="padding:0.35rem 0.9rem;border:none;border-radius:6px;background:#2563eb;color:#fff;cursor:pointer;font-size:0.85rem;font-weight:600;">
                🚀 提交算分
            </button>
        </div>
    </div>
    <div id="kaggle-submit-status" style="font-size:0.85rem;color:var(--text-muted);margin-bottom:0.75rem;"></div>
    <div id="kaggle-history"></div>
</div>

<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.5rem;border-top:1px solid var(--border-color);padding-top:1rem;margin-bottom:0.75rem;">
    <strong style="font-size:1rem;">📋 任务列表</strong>
    <span id="neuro-task-count" style="color:var(--text-muted);font-size:0.8rem;"></span>
</div>
<div id="neuro-task-controls" style="margin-bottom:0.85rem;"></div>
<div id="neuro-tasks-list"></div>
<style>
  .neuro-table { width:100%; border-collapse:collapse; font-size:0.85rem; }
  .neuro-table th { text-align:left; padding:0.45rem 0.7rem; color:var(--text-muted); font-weight:500; border-bottom:1px solid var(--border-color); white-space:nowrap; }
  .neuro-table td { padding:0.4rem 0.7rem; border-bottom:1px solid var(--border-color); white-space:nowrap; }
  .neuro-row:hover { background:var(--bg-card-hover); }
  .neuro-row.clickable { cursor:pointer; }
  .neuro-fbtn { padding:0.25rem 0.6rem; border-radius:6px; font-size:0.78rem; cursor:pointer; border:1px solid var(--border-color); background:var(--bg-card); color:var(--text-primary); }
  .neuro-fbtn.on { background:#2563eb; color:#fff; border-color:#2563eb; }
</style>
`;

let allNeuroTasks = [];
let kaggleSubmissions = [];
let kagglePollTimer = null;
let activeStatus = 'all';
let activeFamily = 'all';
let activeAgent = 'all';
let sortBy = 'id';   // id | score | recent
let kagglePage = 1;
let taskPage = 1;
const KAGGLE_PER_PAGE = 10;
const TASK_PER_PAGE = 20;

// 通用分页控件：当前页/总条数/每页/翻页函数名 -> HTML
function pagerHtml(page, total, perPage, fnName) {
    const pages = Math.max(1, Math.ceil(total / perPage));
    page = Math.min(Math.max(1, page), pages);
    if (pages <= 1) return '';
    const btn = (label, target, disabled) =>
        `<button onclick="${fnName}(${target})" ${disabled ? 'disabled' : ''}
            style="padding:0.22rem 0.6rem;border:1px solid var(--border-color);border-radius:5px;
                   background:var(--bg-card);color:var(--text-primary);font-size:0.78rem;
                   cursor:${disabled ? 'not-allowed' : 'pointer'};opacity:${disabled ? 0.4 : 1};">${label}</button>`;
    return `<div style="display:flex;gap:0.5rem;align-items:center;justify-content:center;margin-top:0.75rem;">
        ${btn('‹ 上一页', page - 1, page <= 1)}
        <span style="color:var(--text-muted);font-size:0.8rem;">第 ${page} / ${pages} 页 · 共 ${total} 条</span>
        ${btn('下一页 ›', page + 1, page >= pages)}
    </div>`;
}
function setKagglePage(p) { kagglePage = p; renderKaggleHistory(); }
function setTaskPage(p) { taskPage = p; renderNeuroTasks(); }

function refreshNeuroGolf(btn) {
    if (btn) { btn.textContent = '⏳'; btn.disabled = true; }
    fetchNeuroGolfTasks(() => {
        if (btn) { btn.textContent = '🔄 刷新'; btn.disabled = false; }
    });
    refreshKaggleSection();
}

function refreshKaggleSection(btn) {
    if (btn) { btn.textContent = '⏳'; btn.disabled = true; }
    // 总是刷新：pending 的补分，complete 的补排名
    const toRefresh = kaggleSubmissions.filter(s => s.status === 'pending' || s.rank == null);
    // 若全都有分有排名，也至少刷新最新一条（用户主动点了就应该查）
    const targets = toRefresh.length > 0 ? toRefresh : kaggleSubmissions.slice(0, 1);
    if (targets.length === 0) { fetchKaggleSubmissions(); if (btn) { btn.textContent = '🔄'; btn.disabled = false; } return; }
    let done = 0;
    targets.forEach(sub => {
        fetch(`/api/project_plugin/neurogolf/kaggle_submissions/${sub.id}/refresh`, {method: 'POST'})
            .then(r => r.json())
            .then(d => { const idx = kaggleSubmissions.findIndex(s => s.id === sub.id); if (idx >= 0) Object.assign(kaggleSubmissions[idx], d); })
            .catch(() => {})
            .finally(() => { if (++done === targets.length) { fetchKaggleSubmissions(); if (btn) { btn.textContent = '🔄'; btn.disabled = false; } } });
    });
}

function fetchNeuroGolfTasks(cb) {
    fetch('/api/project_plugin/neurogolf/status')
        .then(res => res.json())
        .then(data => {
            allNeuroTasks = data.tasks;
            renderFamilyBar();
            renderTaskControls();
            renderNeuroTasks();
            if (cb) cb();
        })
        .catch(err => { console.error("Error fetching neurogolf tasks:", err); if (cb) cb(); });
    fetchKaggleSubmissions();
}


function renderFamilyBar() {
    const bar = document.getElementById('neuro-family-bar');
    if (!bar) return;
    const counts = {}, agentSolved = {};
    allNeuroTasks.forEach(t => {
        counts[t.rule_family] = (counts[t.rule_family] || 0) + 1;
        const author = t.created_by || t.forum?.creator;   // created_by 为空时退回论坛作者
        if (t.status === 'solved' && author) {
            if (!agentSolved[t.rule_family]) agentSolved[t.rule_family] = {};
            agentSolved[t.rule_family][author] = (agentSolved[t.rule_family][author] || 0) + 1;
        }
    });
    const topAgent = (f) => {
        const m = agentSolved[f];
        if (!m) return null;
        const [agent, n] = Object.entries(m).sort((a, b) => b[1] - a[1])[0];
        return {agent, n};
    };
    const families = Object.entries(counts).sort((a, b) => b[1] - a[1]);
    const activeStyle = 'background:#2563eb;color:#fff;border-color:#2563eb;';
    const inactiveStyle = 'background:var(--bg-card);color:var(--text-primary);';
    const btnWrap = (f, label) => {
        const top = topAgent(f);
        const sub = top ? `<div style="font-size:0.7rem;opacity:0.75;margin-top:1px;">${top.agent} ×${top.n}</div>` : '';
        return `<button style="padding:0.3rem 0.7rem;border-radius:8px;font-size:0.8rem;cursor:pointer;border:1px solid var(--border-color);text-align:center;line-height:1.3;${activeFamily === f ? activeStyle : inactiveStyle}" onclick="setNeuroFamily('${f}')">${label}${sub}</button>`;
    };
    bar.innerHTML = btnWrap('all', '全部家族') +
        families.map(([f, n]) => btnWrap(f, `${f} <span style="opacity:0.7">(${n})</span>`)).join('');
}

function setNeuroStatus(status) { activeStatus = status; taskPage = 1; renderTaskControls(); renderNeuroTasks(); }
function setNeuroAgent(agent) { activeAgent = agent; taskPage = 1; renderTaskControls(); renderNeuroTasks(); }
function setNeuroSort(sort) { sortBy = sort; taskPage = 1; renderTaskControls(); renderNeuroTasks(); }

function setNeuroFamily(family) {
    activeFamily = family;
    taskPage = 1;
    renderFamilyBar();
    renderNeuroTasks();
}

function formatAge(sec) {
    if (sec == null) return '—';
    if (sec < 90) return '刚刚';
    const m = Math.floor(sec / 60);
    if (m < 60) return `${m} 分钟前`;
    const h = Math.floor(m / 60);
    if (h < 24) return `${h} 小时前`;
    const d = Math.floor(h / 24);
    return `${d} 天前`;
}

function renderTaskControls() {
    const el = document.getElementById('neuro-task-controls');
    if (!el) return;
    // 提交人列表：来自已完成任务的作者(created_by 回退 forum.creator)，去重
    const agents = [...new Set(allNeuroTasks
        .filter(t => t.status === 'solved')
        .map(t => t.created_by || t.forum?.creator)
        .filter(Boolean))].sort();
    const btn = (on, label, fn) => `<button class="neuro-fbtn${on ? ' on' : ''}" onclick="${fn}">${label}</button>`;
    const group = (title, inner) => `<div style="display:flex;gap:0.3rem;align-items:center;flex-wrap:wrap;">
        <span style="color:var(--text-muted);font-size:0.78rem;margin-right:0.1rem;">${title}</span>${inner}</div>`;
    const statusBtns = [['all', '全部'], ['solved', '✅ 已完成'], ['claimed', '🔧 进行中'], ['open', '⬜ 未完成']]
        .map(([s, l]) => btn(activeStatus === s, l, `setNeuroStatus('${s}')`)).join('');
    const agentBtns = btn(activeAgent === 'all', '全部', `setNeuroAgent('all')`) +
        agents.map(a => btn(activeAgent === a, a, `setNeuroAgent('${a}')`)).join('');
    const sortBtns = [['id', '任务号'], ['score', '最高分'], ['recent', '最近提交']]
        .map(([s, l]) => btn(sortBy === s, l, `setNeuroSort('${s}')`)).join('');
    el.innerHTML = `<div style="display:flex;flex-wrap:wrap;gap:1.2rem;align-items:center;">
        ${group('状态', statusBtns)}${group('提交人', agentBtns)}${group('排序', sortBtns)}</div>`;
}

function fetchKaggleSubmissions() {
    fetch('/api/project_plugin/neurogolf/kaggle_submissions')
        .then(r => r.json())
        .then(d => {
            kaggleSubmissions = d.submissions || [];
            renderKaggleHistory();
            // Auto-poll if latest is pending
            const latest = kaggleSubmissions[0];
            if (latest && latest.status === 'pending') {
                scheduleKagglePoll(latest.id);
            }
        })
        .catch(() => {});
}

function scheduleKagglePoll(subId) {
    if (kagglePollTimer) clearTimeout(kagglePollTimer);
    kagglePollTimer = setTimeout(() => {
        fetch(`/api/project_plugin/neurogolf/kaggle_submissions/${subId}/refresh`, {method: 'POST'})
            .then(r => r.json())
            .then(d => {
                // Update in-memory list
                const idx = kaggleSubmissions.findIndex(s => s.id === subId);
                if (idx >= 0) Object.assign(kaggleSubmissions[idx], d);
                renderKaggleHistory();
                if (d.status === 'pending') scheduleKagglePoll(subId);
            })
            .catch(() => {});
    }, 30000); // 30s
}

function submitToKaggle() {
    const btn = document.getElementById('kaggle-submit-btn');
    const msg = (document.getElementById('kaggle-message')?.value || '').trim() || 'Hub auto submit';
    const statusEl = document.getElementById('kaggle-submit-status');
    btn.disabled = true;
    btn.textContent = '⏳ 提交中...';
    statusEl.textContent = '正在上传 submission.zip 到 Kaggle，请稍候…';

    const fd = new FormData();
    fd.append('message', msg);
    fd.append('submitted_by', 'human');

    fetch('/api/project_plugin/neurogolf/submit', {method: 'POST', body: fd})
        .then(r => r.json().then(d => ({ok: r.ok, d})))
        .then(({ok, d}) => {
            btn.disabled = false;
            btn.textContent = '🚀 提交算分';
            if (!ok) {
                statusEl.innerHTML = `<span style="color:var(--danger,#ef4444)">❌ 提交失败: ${d.detail || JSON.stringify(d)}</span>`;
                return;
            }
            statusEl.innerHTML = `<span style="color:#10b981">✅ 提交成功，已解 ${d.solved_count} 个任务</span>` +
                (d.public_score ? ` — 公榜分: <strong>${d.public_score}</strong>` : ' — 评分中，30s 后自动刷新…');
            fetchKaggleSubmissions();
            if (d.status === 'pending') scheduleKagglePoll(d.id);
        })
        .catch(err => {
            btn.disabled = false;
            btn.textContent = '🚀 提交算分';
            statusEl.innerHTML = `<span style="color:var(--danger,#ef4444)">❌ 网络错误: ${err}</span>`;
        });
}

function renderKaggleHistory() {
    const el = document.getElementById('kaggle-history');
    if (!el) return;
    if (!kaggleSubmissions.length) {
        el.innerHTML = '<div style="color:var(--text-muted);font-size:0.8rem;">暂无历史提交记录</div>';
        return;
    }
    const kPages = Math.max(1, Math.ceil(kaggleSubmissions.length / KAGGLE_PER_PAGE));
    if (kagglePage > kPages) kagglePage = kPages;
    const pageItems = kaggleSubmissions.slice((kagglePage - 1) * KAGGLE_PER_PAGE, kagglePage * KAGGLE_PER_PAGE);
    const rows = pageItems.map(s => {
        const scoreCell = s.public_score != null
            ? `<strong>${s.public_score.toFixed(3)}</strong>`
            : (s.status === 'pending' ? '<span style="color:var(--text-muted)">评分中…</span>' : '—');
        const rankCell = s.rank ? `#${s.rank}${s.total_teams ? ' / ' + s.total_teams : ''}` : '—';
        const statusBadge = s.status === 'complete'
            ? `<span style="color:#10b981">●</span>`
            : s.status === 'pending'
            ? `<span style="color:#f59e0b">●</span>`
            : `<span style="color:#ef4444">●</span>`;
        const dt = s.submitted_at ? s.submitted_at.replace('T',' ').slice(0,16) : '—';
        return `<tr style="border-top:1px solid var(--border-color);">
            <td style="padding:0.4rem 0.6rem;white-space:nowrap;">${statusBadge} ${dt}</td>
            <td style="padding:0.4rem 0.6rem;">${s.message || '—'}</td>
            <td style="padding:0.4rem 0.6rem;text-align:center;">${s.solved_count ?? '—'}</td>
            <td style="padding:0.4rem 0.6rem;text-align:center;">${scoreCell}</td>
            <td style="padding:0.4rem 0.6rem;text-align:center;">${rankCell}</td>
            <td style="padding:0.4rem 0.6rem;text-align:center;">${s.submitted_by || '—'}</td>
        </tr>`;
    }).join('');
    el.innerHTML = `<table style="width:100%;font-size:0.8rem;border-collapse:collapse;">
        <thead><tr style="color:var(--text-muted);">
            <th style="padding:0.3rem 0.6rem;text-align:left;font-weight:500;">时间</th>
            <th style="padding:0.3rem 0.6rem;text-align:left;font-weight:500;">备注</th>
            <th style="padding:0.3rem 0.6rem;text-align:center;font-weight:500;">已解</th>
            <th style="padding:0.3rem 0.6rem;text-align:center;font-weight:500;">公榜分</th>
            <th style="padding:0.3rem 0.6rem;text-align:center;font-weight:500;">排名</th>
            <th style="padding:0.3rem 0.6rem;text-align:center;font-weight:500;">提交人</th>
        </tr></thead>
        <tbody>${rows}</tbody>
    </table>` + pagerHtml(kagglePage, kaggleSubmissions.length, KAGGLE_PER_PAGE, 'setKagglePage');
}

function renderNeuroTasks() {
    const grid = document.getElementById('neuro-tasks-list');
    if (!grid) return;

    let filtered = allNeuroTasks;
    if (activeStatus !== 'all') filtered = filtered.filter(t => t.status === activeStatus);
    if (activeFamily !== 'all') filtered = filtered.filter(t => t.rule_family === activeFamily);
    if (activeAgent !== 'all') filtered = filtered.filter(t => (t.created_by || t.forum?.creator) === activeAgent);

    // KPI 跟随家族过滤（状态过滤不影响 KPI 数字）
    const kpiBase = activeFamily === 'all' ? allNeuroTasks : allNeuroTasks.filter(t => t.rule_family === activeFamily);
    const solvedTasks = kpiBase.filter(t => t.status === 'solved');
    const solvedCount = solvedTasks.length;
    const unledgered = solvedTasks.filter(t => !t.ledger_verified).length;
    const claimedTasks = kpiBase.filter(t => t.status === 'claimed');
    const claimedCount = claimedTasks.length;
    const openCount = kpiBase.filter(t => t.status === 'open').length;

    // 各 AI 分项：已完成按作者(created_by 回退 forum.creator)，进行中按认领人。
    // 全部基于 kpiBase，所以自动跟随家族过滤。
    const byAgent = (tasks, getAgent) => {
        const m = {};
        tasks.forEach(t => { const a = getAgent(t); if (a) m[a] = (m[a] || 0) + 1; });
        return Object.entries(m).sort((a, b) => b[1] - a[1]);
    };
    const solvedByAgent = byAgent(solvedTasks, t => t.created_by || t.forum?.creator);
    const claimedByAgent = byAgent(claimedTasks, t => t.claim?.agent);
    const agentBreakdownHtml = (entries, accent) => entries.length ? `
        <div style="display:flex;flex-direction:column;gap:0.2rem;font-size:0.72rem;
                    border-left:1px solid var(--border-color);padding-left:0.7rem;align-self:stretch;justify-content:center;">
            ${entries.map(([a, n]) => `<div style="display:flex;justify-content:space-between;gap:0.7rem;white-space:nowrap;">
                <span style="color:var(--text-muted);max-width:90px;overflow:hidden;text-overflow:ellipsis;">${a}</span>
                <span style="font-weight:700;color:${accent};font-family:var(--mono);">${n}</span>
            </div>`).join('')}
        </div>` : '';

    const unledgeredTasks = solvedTasks.filter(t => !t.ledger_verified);
    const unledgeredRows = unledgeredTasks.map(t => {
        const owner = t.created_by || t.forum?.creator || '—';
        return `<tr><td style="padding:0.2rem 0.5rem;white-space:nowrap;">${t.id}</td><td style="padding:0.2rem 0.5rem;color:#9ca3af;">${t.rule_family}</td><td style="padding:0.2rem 0.5rem;font-weight:600;color:#f59e0b;">${owner}</td></tr>`;
    }).join('');
    const unledgeredHtml = unledgered ? `
        <span style="position:relative;display:inline-block;">
            <span style="font-size:0.75rem;color:var(--text-muted);cursor:default;border-bottom:1px dotted var(--text-muted);"
                  class="unledgered-trigger">其中 ${unledgered} 个未过账</span>
            <div style="display:none;position:fixed;
                        background:#1e2130;border:1px solid #374151;border-radius:8px;padding:0.5rem;
                        min-width:220px;z-index:9999;box-shadow:0 4px 16px rgba(0,0,0,0.4);font-size:0.78rem;"
                 class="unledgered-popup">
                <div style="color:#9ca3af;margin-bottom:0.3rem;font-size:0.72rem;">task · 家族 · 负责 AI</div>
                <table style="border-collapse:collapse;width:100%;">${unledgeredRows}</table>
            </div>
        </span>` : '';

    document.getElementById('neuro-kpis').innerHTML = `
        <div class="kpi-card" style="--accent: #10b981;">
            <span class="kpi-title">✅ 已完成</span>
            <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:0.6rem;flex-wrap:wrap;">
                <div style="display:flex;flex-direction:column;">
                    <span class="kpi-value">${solvedCount} / ${kpiBase.length}</span>
                    ${unledgeredHtml}
                </div>
                ${agentBreakdownHtml(solvedByAgent, '#10b981')}
            </div>
        </div>
        <div class="kpi-card" style="--accent: #f59e0b;">
            <span class="kpi-title">🔧 进行中 (已认领)</span>
            <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:0.6rem;flex-wrap:wrap;">
                <span class="kpi-value">${claimedCount}</span>
                ${agentBreakdownHtml(claimedByAgent, '#f59e0b')}
            </div>
        </div>
        <div class="kpi-card" style="--accent: #6b7280;">
            <span class="kpi-title">⬜ 未完成</span>
            <span class="kpi-value">${openCount}</span>
        </div>
    `;

    // 排序
    const sorted = [...filtered].sort((a, b) => {
        if (sortBy === 'score') {
            const sa = a.best_score == null ? -Infinity : a.best_score;
            const sb = b.best_score == null ? -Infinity : b.best_score;
            return sb - sa;                       // 高分在前
        }
        if (sortBy === 'recent') {
            const aa = a.artifact_age == null ? Infinity : a.artifact_age;
            const bb = b.artifact_age == null ? Infinity : b.artifact_age;
            return aa - bb;                       // age 越小越新, 最近在前
        }
        return a.id.localeCompare(b.id);          // 任务号
    });

    const countEl = document.getElementById('neuro-task-count');
    if (countEl) countEl.textContent = `显示 ${sorted.length} / ${allNeuroTasks.length} 个任务`;

    if (!sorted.length) {
        grid.innerHTML = `<div style="color:var(--text-muted);font-size:0.85rem;padding:1rem;">无匹配任务</div>`;
        return;
    }

    const tPages = Math.max(1, Math.ceil(sorted.length / TASK_PER_PAGE));
    if (taskPage > tPages) taskPage = tPages;
    const pageItems = sorted.slice((taskPage - 1) * TASK_PER_PAGE, taskPage * TASK_PER_PAGE);

    const rows = pageItems.map(t => {
        const st = t.status === 'solved'
            ? { icon: '✅', text: t.ledger_verified ? '已完成' : '已完成·未过账', color: '#10b981' }
            : t.status === 'claimed'
            ? { icon: '🔧', text: '进行中', color: '#f59e0b' }
            : { icon: '⬜', text: '未完成', color: 'var(--text-muted)' };
        const score = t.best_score != null
            ? `<span style="color:#f59e0b;font-weight:600;">${t.best_score.toFixed(3)}</span>`
            : '<span style="color:var(--text-muted)">—</span>';
        const submitter = t.created_by || t.forum?.creator
            || (t.status === 'claimed' && t.claim ? `🔧 ${t.claim.agent}` : '') || '—';
        const topic = t.forum
            ? `<span style="color:#60a5fa;">#${t.forum.topic_id}</span>`
            : '<span style="color:var(--text-muted)">—</span>';
        const onclick = t.forum ? `onclick="expandTopic(${t.forum.topic_id})"` : '';
        return `<tr class="neuro-row${t.forum ? ' clickable' : ''}" ${onclick}>
            <td style="font-weight:600;">${t.id}</td>
            <td><span class="topic-tag">${t.rule_family}</span></td>
            <td style="color:${st.color};">${st.icon} ${st.text}</td>
            <td style="text-align:right;font-family:var(--mono);">${score}</td>
            <td>${submitter}</td>
            <td style="color:var(--text-muted);">${formatAge(t.artifact_age)}</td>
            <td>${topic}</td>
        </tr>`;
    }).join('');

    grid.innerHTML = `<table class="neuro-table">
        <thead><tr>
            <th>任务</th><th>家族</th><th>状态</th>
            <th style="text-align:right;">最高分</th><th>提交人</th><th>更新</th><th>帖子</th>
        </tr></thead>
        <tbody>${rows}</tbody>
    </table>` + pagerHtml(taskPage, sorted.length, TASK_PER_PAGE, 'setTaskPage');

    // 未过账悬浮框
    document.querySelectorAll('.unledgered-trigger').forEach(trigger => {
        const popup = trigger.parentElement.querySelector('.unledgered-popup');
        if (!popup) return;
        // position:fixed + 动态定位, 才能逃出 .kpi-card 的 overflow:hidden 裁切
        trigger.addEventListener('mouseenter', () => {
            popup.style.display = 'block';            // 先显示才能量到尺寸
            const r = trigger.getBoundingClientRect();
            const pw = popup.offsetWidth, ph = popup.offsetHeight;
            let left = r.left + r.width / 2 - pw / 2;
            left = Math.max(8, Math.min(left, window.innerWidth - pw - 8));
            let top = r.top - ph - 6;                 // 默认放上方
            if (top < 8) top = r.bottom + 6;          // 上方放不下就放下方
            popup.style.left = left + 'px';
            popup.style.top = top + 'px';
        });
        trigger.addEventListener('mouseleave', () => { popup.style.display = 'none'; });
    });
}

window.registerPluginView('view-neurogolf-tasks', '⛳', '400 任务追踪', neurogolfHtml, fetchNeuroGolfTasks);
