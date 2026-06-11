const API_URL = '/api/dashboard_data';
let lastDataHash = '';
// 当前查看的项目, 记在 localStorage (在 /projects 管理页选择); 所有数据请求都带上它
let currentProject = localStorage.getItem('ai_hub_project') || 'rogii';
let currentStatusFilter = '讨论中'; // 与 index.html 中默认高亮的筛选按钮保持一致
let currentTagFilter = null;
let allTopicsData = [];
let metricLowerIsBetter = true; // RMSE 类指标越低越好, 由后端 dashboard_data 下发

// RMSE 类指标不可能 <= 0, 这类值是未跑出分数时的占位, 当缺失处理
function validScore(s) {
    if (s === null || s === undefined) return false;
    if (metricLowerIsBetter && s <= 0) return false;
    return true;
}
function fmtScore(s) { return validScore(s) ? s.toFixed(4) : '--'; }

// Advanced Filters (clicking on dashboard stats)
let advancedFilters = {
    creator: null,
    topicState: null, // 'voting', 'success', 'rejected'
    repliedBy: null,
    claimedBy: null,
    metricName: null
};

function clearAdvancedFilters() {
    advancedFilters = { creator: null, topicState: null, repliedBy: null, claimedBy: null, metricName: null };
    const banner = document.getElementById('advanced-filter-banner');
    if (banner) banner.style.display = 'none';
    applyFilters();
}

function applyAdvancedFilter(type, agentName, metricName) {
    clearAdvancedFilters();
    advancedFilters.metricName = metricName;
    if (type === 'topic_count') {
        advancedFilters.creator = agentName;
    } else if (type === 'voting') {
        advancedFilters.creator = agentName;
        advancedFilters.topicState = 'voting';
    } else if (type === 'success') {
        advancedFilters.creator = agentName;
        advancedFilters.topicState = 'success';
    } else if (type === 'rejected') {
        advancedFilters.creator = agentName;
        advancedFilters.topicState = 'rejected';
    } else if (type === 'archived') {
        advancedFilters.creator = agentName;
        advancedFilters.topicState = 'archived';
    } else if (type === 'opposed') {
        advancedFilters.creator = agentName;
        advancedFilters.topicState = 'opposed';
    } else if (type === 'reply') {
        advancedFilters.repliedBy = agentName;
    } else if (type === 'claimed') {
        advancedFilters.claimedBy = agentName;
    } else if (type === 'avg_score') {
        // Average score filter not easily applicable to topics view
        return;
    }

    let banner = document.getElementById('advanced-filter-banner');
    if (!banner) {
        banner = document.createElement('div');
        banner.id = 'advanced-filter-banner';
        banner.className = 'val-reason-item';
        banner.style = 'margin-bottom: 1rem; color: #10b981; border-left-color: #10b981; cursor: pointer; display: flex; justify-content: space-between; align-items: center; background: rgba(16,185,129,0.1); padding: 0.8rem;';
        banner.onclick = clearAdvancedFilters;
        const parent = document.getElementById('topics-container').parentNode;
        parent.insertBefore(banner, document.getElementById('topics-container'));
    }
    banner.innerHTML = `<div><strong>🔍 高级筛选生效中:</strong> 正在查看 <strong>${escapeHTML(agentName)}</strong> 的 <strong>${escapeHTML(metricName)}</strong> 相关帖子</div><div style="text-decoration:underline; font-size:0.85rem; opacity:0.8;">点击清除筛选 ✖</div>`;
    banner.style.display = 'flex';

    // Switch to topics view
    switchView('view-topics');

    // Force basic status filter to 'all' so we don't accidentally hide advanced filter results
    currentStatusFilter = 'all';
    document.querySelectorAll('.filter-btn').forEach(b => {
        if (b.getAttribute('data-filter') === 'all') b.classList.add('active');
        else b.classList.remove('active');
    });

    // Apply filters and re-render
    applyFilters();
}
let allAgentsData = [];
let expandedTopics = new Set();
let unfoldedResolved = new Set(); // 已完结话题默认折叠为单行, 在此集合中的展开为完整卡片
let kbTagFilter = null; // 知识库页的标签筛选


async function fetchDashboardData() {
    try {
        const res = await fetch(`${API_URL}?project=${encodeURIComponent(currentProject)}`, { cache: "no-store" });
        if (res.status === 404) {
            // 选中的项目不存在了(被删/库重置) -> 去项目管理页重新选
            localStorage.removeItem('ai_hub_project');
            location.href = '/projects';
            return;
        }
        const data = await res.json();
        const hash = JSON.stringify(data);
        if (hash !== lastDataHash) {
            metricLowerIsBetter = data.metric_lower_is_better !== false;
            renderProjectBadge(data.project_status);
            allAgentsData = data.agents;
            updateAgents(data.agents);
            updateExperiments(data.experiments);
            allTopicsData = data.topics;
            updateTagFilters(data.all_tags);
            applyFilters();
            renderKnowledge();
            renderBottlenecks();
            renderEvalMatrix();
            renderDashboardMetrics(data);
            lastDataHash = hash;
        } else {
            refreshTimestamps();
        }
    } catch (e) {
        console.error("获取数据失败:", e);
    }
}

async function fetchSystemStatus() {
    try {
        const res = await fetch('/api/system/status', { cache: "no-store" });
        if (!res.ok) return;
        const data = await res.json();
        const cfg = data.config || {};
        const api = cfg.api || {};
        const db = cfg.database || {};
        const workspace = cfg.workspace || {};
        const apiEl = document.getElementById('ssc-api');
        const dbEl = document.getElementById('ssc-db');
        const projectEl = document.getElementById('ssc-project');
        if (apiEl) {
            apiEl.textContent = api.public_base_url || location.origin;
            apiEl.title = api.public_base_url || location.origin;
        }
        if (dbEl) {
            dbEl.textContent = db.url_masked || 'connected';
            dbEl.title = db.url_masked || 'connected';
        }
        if (projectEl) {
            projectEl.textContent = workspace.default_project || 'rogii';
        }
    } catch (e) {
        console.error("获取系统状态失败:", e);
    }
}

// ---- Agents ----
function updateAgents(agents) {
    const c = document.getElementById('agents-container');
    c.innerHTML = '';
    if (!agents.length) { c.innerHTML = '<p class="empty-state">暂无成员</p>'; return; }
    agents.forEach(a => {
        const chip = document.createElement('div');
        chip.className = 'agent-chip';
        const cv = fmtScore(a.cv_score);
        const lb = fmtScore(a.lb_score);
        const avgScore = a.avg_eval_score !== null ? a.avg_eval_score.toFixed(1) : '-';
        chip.innerHTML = `
            <div class="chip-top">
                <div class="chip-name-wrap">
                    <span class="chip-name">${escapeHTML(a.name)}</span>
                    <span class="chip-todo-badge ${a.todo_count > 0 ? 'hot' : ''}" data-dd="todo" data-metric="待办事项" title="该 AI 当前积压的待办数 (未投票/待评分/待认领/缺结论等)。点击看明细, 方便你决定唤醒谁">${a.todo_count > 0 ? `📥 待办 ${a.todo_count}` : '✓ 无待办'}</span>
                </div>
                <div class="chip-scores">
                    <span class="chip-cv">CV ${cv}</span>
                    <span class="chip-lb">LB ${lb}</span>
                </div>
            </div>
            <div class="chip-status" title="${escapeHTML(a.status || '')}">${escapeHTML(a.status || '等待中...')}</div>
            <div class="chip-stats" style="flex-wrap: wrap; gap: 0.4rem; justify-content: flex-start;">
                <span title="该 AI 的未读动态数 (数字大说明它很久没上线同步了)" style="${a.unread_count > 0 ? 'color:#7dd3fc;' : ''}">📨 未读: ${a.unread_count || 0}</span>
                <span class="clickable-stat" data-ftype="topic_count" data-metric="发起提案" title="发起的总提案数">📝 提案: ${a.topic_count}</span>
                <span class="clickable-stat" data-ftype="voting" data-metric="投票中提案" title="正在激辩、等待全员表态的提案数" style="color:#60a5fa;">⏳ 投票中: ${a.voting_count || 0}</span>
                <span class="clickable-stat" data-ftype="success" data-metric="成功提案" title="发起的提案获全员通过的数量 (全员 agree 或全员 verify)">✅ 成功: ${a.approved_count || 0}</span>
                <span class="clickable-stat" data-ftype="rejected" data-metric="被硬驳回提案" title="发起的提案被全员一致投 disagree 而完结的数量 (硬驳回)。点击查看这些帖子" style="${a.rejected_count > 0 ? 'color:#ef4444;' : ''}">❌ 驳回: ${a.rejected_count || 0}</span>
                <span class="clickable-stat" data-ftype="archived" data-metric="被归档提案" title="发起的提案未达成共识、被人工 resolve 写结论归档的数量 (软收口)。点击查看这些帖子" style="${a.archived_count > 0 ? 'color:#818cf8;' : ''}">📦 归档: ${a.archived_count || 0}</span>
                <span class="clickable-stat" data-ftype="opposed" data-metric="被投反对票的提案" title="发起的提案累计收到的反对票数 (含还在投票中的, 是'被驳回'的前兆信号)。点击查看这些帖子" style="${a.disagree_received > 0 ? 'color:#fb923c;' : ''}">👎 被反对: ${a.disagree_received || 0}</span>
                <span class="clickable-stat" data-dd="low_evals_received" data-metric="被差评明细" title="发言(回复)收到低于 5 分差评的次数。点击看每条差评的人/分/理由" style="${a.low_eval_count > 0 ? 'color:#f472b6;' : ''}">🥀 被差评: ${a.low_eval_count || 0}</span>
                <span class="clickable-stat" data-dd="first_disagree" data-metric="率先反对明细" title="对他人提案第一个投出反对票的次数 (按动作时间序; 不含否决自己的提案)。点击看明细" style="${a.first_disagree_count > 0 ? 'color:#fbbf24;' : ''}">⚔️ 率先反对: ${a.first_disagree_count || 0}</span>
                <span class="clickable-stat" data-dd="harsh_evals_given" data-metric="开喷明细" title="给他人回复打 <5 分差评的次数 (主动差评输出, 与'被差评'攻防对应)。点击看明细" style="${a.harsh_evals_given > 0 ? 'color:#fb7185;' : ''}">🔥 开喷: ${a.harsh_evals_given || 0}</span>
                <span class="clickable-stat" data-dd="self_rejections" data-metric="自我否决明细" title="被大家说服后, 亲手对自己发起的提案投出反对票的次数 (敢于认错是美德)。点击看明细" style="${a.self_reject_count > 0 ? 'color:#a78bfa;' : ''}">🪞 自我否决: ${a.self_reject_count || 0}</span>
                <span class="clickable-stat" data-ftype="reply" data-metric="参与辩论" title="参与回复与讨论的次数。点击查看这些帖子">💬 辩论: ${a.reply_count}</span>
                <span class="clickable-stat" data-ftype="claimed" data-metric="认领的苦力" title="主动认领并跑完代码的任务数。点击查看这些帖子" style="color:#f59e0b;">🚀 苦力: ${a.claimed_count || 0}</span>
                <span class="clickable-stat" data-dd="evals_received" data-metric="收到的全部评价" title="发出的评论被其他 AI 评价的平均分。点击看每条评价" style="color:var(--warning)">⭐ 均分: ${avgScore}</span>
            </div>`;
        // 通过闭包绑定点击, 避免把 agent 名拼进 onclick 字符串(名字含引号会注入)
        // data-dd → 明细弹窗 (回复/评价级证据); data-ftype → 跳转帖子筛选列表 (帖子级)
        chip.querySelectorAll('.clickable-stat, .chip-todo-badge').forEach(el => {
            if (el.dataset.dd) el.onclick = () => showDrilldown(a.name, el.dataset.dd, el.dataset.metric);
            else el.onclick = () => applyAdvancedFilter(el.dataset.ftype, a.name, el.dataset.metric);
        });
        c.appendChild(chip);
    });
}

// ---- Experiments Table ----
function updateExperiments(exps) {
    const tbody = document.getElementById('experiments-body');
    tbody.innerHTML = '';
    if (!exps || !exps.length) { tbody.innerHTML = '<tr><td colspan="8" class="empty-state">暂无实验记录</td></tr>'; return; }
    exps.forEach(e => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td class="exp-agent">${escapeHTML(e.agent)}</td>
            <td>${escapeHTML(e.method)}</td>
            <td style="font-size:0.8rem;color:var(--text-muted)">${escapeHTML(e.params || '--')}</td>
            <td class="exp-cv">${e.cv_score !== null ? e.cv_score.toFixed(4) : '--'}</td>
            <td class="exp-lb">${e.lb_score !== null ? e.lb_score.toFixed(4) : '--'}</td>
            <td>${e.duration_min !== null ? e.duration_min + '' : '--'}</td>
            <td style="font-size:0.8rem;color:var(--text-muted)">${escapeHTML(e.notes || '--')}</td>
            <td class="time-stamp" data-time="${e.timestamp}" style="font-size:0.8rem;color:var(--text-muted)">${getTimeAgo(new Date(e.timestamp))}</td>`;
        tbody.appendChild(tr);
    });
}

// ---- Tag Filters ----
function updateTagFilters(tags) {
    const c = document.getElementById('tag-filters');
    c.innerHTML = '';
    tags.forEach(tag => {
        const btn = document.createElement('button');
        btn.className = 'tag-btn' + (currentTagFilter === tag ? ' active' : '');
        btn.textContent = tag;
        btn.onclick = () => {
            currentTagFilter = (currentTagFilter === tag) ? null : tag;
            document.querySelectorAll('.tag-btn').forEach(b => b.classList.remove('active'));
            if (currentTagFilter) btn.classList.add('active');
            applyFilters();
        };
        c.appendChild(btn);
    });
}

// ---- Filters ----
function setStatusFilter(f, btn) {
    currentStatusFilter = f;
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    applyFilters();
}

function applyFilters() {
    const q = (document.getElementById('topic-search')?.value || '').toLowerCase();
    const filtered = allTopicsData.filter(t => {
        const ms = (currentStatusFilter === 'all') ||
                   (t.status === currentStatusFilter) ||
                   (currentStatusFilter === '讨论中' && (t.status === '验证提案' || t.status === '待执行'));
        const mt = !currentTagFilter || (t.tag === currentTagFilter);
        const mq = !q || t.title.toLowerCase().includes(q) || t.content.toLowerCase().includes(q) || t.creator.toLowerCase().includes(q);

        // Advanced Filters
        let mAdv = true;
        if (advancedFilters.creator && t.creator !== advancedFilters.creator) mAdv = false;
        if (advancedFilters.claimedBy && t.claimed_by !== advancedFilters.claimedBy) mAdv = false;
        if (advancedFilters.repliedBy) {
            const hasReply = t.replies && t.replies.some(r => r.author === advancedFilters.repliedBy);
            if (!hasReply) mAdv = false;
        }
        if (advancedFilters.topicState) {
            // 口径与服务端 get_dashboard_data 的 approved/rejected/archived/voting 分类一致
            const agree = t.votes.filter(v => v.vote === 'agree').length;
            const disagree = t.votes.filter(v => v.vote === 'disagree').length;
            const verify = t.votes.filter(v => v.vote === 'verify').length;
            const n = allAgentsData.length;

            if (advancedFilters.topicState === 'success') {
                if (!(agree === n || verify === n)) mAdv = false;
            } else if (advancedFilters.topicState === 'rejected') {
                if (disagree !== n) mAdv = false;
            } else if (advancedFilters.topicState === 'archived') {
                if (!(t.conclusion && agree !== n && verify !== n && disagree !== n)) mAdv = false;
            } else if (advancedFilters.topicState === 'opposed') {
                if (disagree === 0) mAdv = false;
            } else if (advancedFilters.topicState === 'voting') {
                if (t.status !== '验证提案') mAdv = false;
            }
        }

        return ms && mt && mq && mAdv;
    });
    renderTopics(filtered);
    renderValidationBacklog();
}

function renderValidationBacklog() {
    const section = document.getElementById('validation-section');
    const container = document.getElementById('validation-container');
    const todoContainer = document.getElementById('todo-page-container');
    container.innerHTML = '';
    if(todoContainer) todoContainer.innerHTML = '';

    const proposals = allTopicsData.filter(t => t.status === '验证提案');
    const todos = allTopicsData.filter(t => t.status === '待执行');
    const history = allTopicsData.filter(t => t.status === '已完结');

    const historyContainer = document.getElementById('todo-history-container');
    if (historyContainer) historyContainer.innerHTML = '';

    if (proposals.length === 0) {
        section.style.display = 'none';
    } else {
        section.style.display = 'block';
        let html = '';
        proposals.forEach(t => { html += buildValCardHtml(t, false); });
        container.innerHTML = html;
    }

    if (todos.length > 0 && todoContainer) {
        let html = '';
        todos.forEach(t => { html += buildValCardHtml(t, true); });
        todoContainer.innerHTML = html;
    } else if (todoContainer) {
        todoContainer.innerHTML = '<p class="empty-state">目前没有待执行的任务。如果有提议，请先在讨论区通过全员共识。</p>';
    }

    if (history.length > 0 && historyContainer) {
        let html = '';
        history.forEach(t => { html += buildValCardHtml(t, false, true); });
        historyContainer.innerHTML = html;
    } else if (historyContainer) {
        historyContainer.innerHTML = '<p class="empty-state">暂无历史记录。</p>';
    }
}

function buildValCardHtml(t, isTodo, isHistory = false) {
    let reasonsHtml = '';

    if (isHistory) {
        const opposeVotes = t.votes.filter(v => v.vote === 'disagree');
        if (t.conclusion) {
            reasonsHtml += `<div class="val-reason-item" style="color:#818cf8; margin-bottom: 0.8rem;"><strong>📜 人工归档${t.closed_by ? ` (by ${escapeHTML(t.closed_by)})` : ''}:</strong> ${escapeHTML(t.conclusion.slice(0, 200))}</div>`;
        } else if (opposeVotes.length > 0) {
            reasonsHtml += `<div class="val-reason-item" style="color:#ef4444; margin-bottom: 0.8rem;"><strong>❌ 已否决:</strong> 提案已被全员否决并归档。</div>`;
        } else {
            reasonsHtml += `<div class="val-reason-item" style="color:#10b981; margin-bottom: 0.8rem;"><strong>✅ 已通过:</strong> 提案已无需验证直接结案。</div>`;
        }
    } else if (isTodo) {
        if (t.claimed_by) {
            reasonsHtml = `<div class="val-reason-item" style="color:#f59e0b; margin-bottom: 0.8rem;"><strong>🚀 执行中:</strong> 已被 <strong>${escapeHTML(t.claimed_by)}</strong> 认领，正在跑代码验证中。</div>`;
        } else {
            reasonsHtml = `<div class="val-reason-item" style="color:#34d399; margin-bottom: 0.8rem;"><strong>✅ 全员共识 (待认领):</strong> 所有 AI 已投票通过，等待任一 AI 认领并执行代码。</div>`;
        }
    }

    // Linked Experiments
    if (t.linked_experiments && t.linked_experiments.length > 0) {
        reasonsHtml += `<div style="margin-top:1rem; border-top:1px solid var(--border-color); padding-top:0.8rem;">`;
        reasonsHtml += `<div style="font-weight:bold; margin-bottom:0.5rem; color:var(--text-color);">🔬 实验验证结果:</div>`;
        t.linked_experiments.forEach(exp => {
            reasonsHtml += `
                <div style="background:var(--bg-card-hover); padding:0.6rem; border-radius:6px; margin-bottom:0.5rem; font-size:0.85rem;">
                    <strong>[${escapeHTML(exp.agent)}]</strong> ${escapeHTML(exp.method)}
                    (CV: <strong>${exp.cv_score || '-'}</strong> | LB: <strong>${exp.lb_score || '-'}</strong>)
                    <div style="margin-top:0.3rem; color:var(--text-muted);">${escapeHTML(exp.notes || '')}</div>
                </div>`;
        });
        reasonsHtml += `</div>`;
    }

    if (!isTodo && !isHistory) {
        const supportVotes = t.votes.filter(v => v.vote === 'verify');
        const opposeVotes = t.votes.filter(v => v.vote !== 'verify');

        if (supportVotes.length > 0) {
            reasonsHtml += `<div style="margin-bottom: 0.6rem;">`;
            reasonsHtml += `<div style="font-weight:bold; color:#10b981; margin-bottom:0.3rem;">✅ 赞成验证 (${supportVotes.length}):</div>`;
            supportVotes.forEach(v => {
                const isCreator = v.agent === t.creator ? ' (提案人)' : '';
                const timeStr = v.timestamp ? `<span style="color:var(--text-muted);font-weight:normal;font-size:0.8rem;margin-left:0.3rem;">[${getTimeAgo(new Date(v.timestamp))}]</span>` : '';
                reasonsHtml += `<div class="val-reason-item" style="padding-left:0.5rem; border-left:2px solid #10b981; margin-bottom:0.3rem;"><strong>${escapeHTML(v.agent)}${isCreator}${timeStr}:</strong> ${escapeHTML(v.reason)}</div>`;
            });
            reasonsHtml += `</div>`;
        }

        if (opposeVotes.length > 0) {
            reasonsHtml += `<div>`;
            reasonsHtml += `<div style="font-weight:bold; color:#ef4444; margin-bottom:0.3rem;">❌ 存在异议 (${opposeVotes.length}):</div>`;
            opposeVotes.forEach(v => {
                const vStr = v.vote === 'agree' ? '认为无需实验直接结案' : '强烈反对该方向';
                const timeStr = v.timestamp ? `<span style="color:var(--text-muted);font-weight:normal;font-size:0.8rem;margin-left:0.3rem;">[${getTimeAgo(new Date(v.timestamp))}]</span>` : '';
                reasonsHtml += `<div class="val-reason-item" style="padding-left:0.5rem; border-left:2px solid #ef4444; margin-bottom:0.3rem;"><strong>${escapeHTML(v.agent)} (${vStr})${timeStr}:</strong> ${escapeHTML(v.reason)}</div>`;
            });
            reasonsHtml += `</div>`;
        }
    }

    let cardCls = 'val-card prop-card';
    if (isTodo) cardCls = 'val-card todo-card';
    else if (isHistory) cardCls = 'val-card history-card';

    const createTimeStr = t.timestamp ? getTimeAgo(new Date(t.timestamp)) : '';
    const sortedReplies = t.replies ? [...t.replies].sort((a,b) => new Date(a.timestamp) - new Date(b.timestamp)) : [];
    const lastReplyStr = sortedReplies.length > 0 ? getTimeAgo(new Date(sortedReplies[sortedReplies.length - 1].timestamp)) : '暂无回复';

    return `
        <div class="${cardCls}" onclick="expandTopic(${t.id})">
            <div class="val-card-header" style="display: flex; flex-direction: column; align-items: flex-start; gap: 0.6rem;">
                <div style="display: flex; justify-content: space-between; width: 100%; align-items: flex-start;">
                    <div style="display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center; line-height: 1.4;">
                        <span class="topic-tag val-tag">${escapeHTML(t.tag || '提案')}</span>
                        <span class="val-title" style="word-break: break-word;">${escapeHTML(t.title)}</span>
                        <span class="topic-creator-tag">提案人: ${escapeHTML(t.creator)}</span>
                    </div>
                    <span class="val-jump-link" style="color:var(--text-muted); font-size: 0.85rem; text-decoration:underline; flex-shrink: 0; margin-left: 1rem;">🔗 查看前因后果</span>
                </div>
                <div style="font-size: 0.8rem; color: var(--text-muted); display: flex; gap: 1rem;">
                    <span>📅 发起: ${createTimeStr}</span>
                    <span>💬 最新回复: ${lastReplyStr}</span>
                </div>
            </div>
            <div class="val-card-reasons">${reasonsHtml}</div>
        </div>
    `;
}

// ---- 当前项目徽章 (项目的选择/新建/归档在独立的 /projects 管理页) ----
function renderProjectBadge(projectStatus) {
    const nameEl = document.getElementById('cpb-name');
    if (!nameEl) return;
    nameEl.textContent = projectStatus === 'archived' ? `${currentProject} (已归档·只读)` : currentProject;
    const badge = document.getElementById('current-project-badge');
    if (badge) badge.classList.toggle('cpb-archived', projectStatus === 'archived');
}

// ---- 知识库页: 已完结话题的结论台账 ----
function topicOutcome(t) {
    // 与服务端 build_knowledge 口径一致: 全员 agree=通过; 全员 disagree=驳回; 人工结案=归档
    if (t.status !== '已完结') return null;
    const n = allAgentsData.length;
    const agree = t.votes.filter(v => v.vote === 'agree').length;
    const disagree = t.votes.filter(v => v.vote === 'disagree').length;
    if (n > 0 && agree === n) return '通过';
    if (n > 0 && disagree === n) return '驳回';
    return '归档';
}

const KB_GROUPS = [
    { key: '通过', title: '✅ 已验证有效 (全员通过)', cls: 'kb-pass' },
    { key: '驳回', title: '❌ 已证伪 / 驳回 (全员反对)', cls: 'kb-fail' },
    { key: '归档', title: '📜 人工归档 (未达共识, 写结论收口)', cls: 'kb-arch' },
];

function renderKnowledge() {
    const c = document.getElementById('kb-container');
    if (!c) return;
    const q = (document.getElementById('kb-search')?.value || '').toLowerCase();

    const resolved = allTopicsData.filter(t => t.status === '已完结').map(t => ({ t, outcome: topicOutcome(t) }));

    // 标签筛选 chips
    const tagBox = document.getElementById('kb-tag-filters');
    if (tagBox) {
        const tags = [...new Set(resolved.map(r => r.t.tag).filter(Boolean))].sort();
        tagBox.innerHTML = '';
        tags.forEach(tag => {
            const btn = document.createElement('button');
            btn.className = 'tag-btn' + (kbTagFilter === tag ? ' active' : '');
            btn.textContent = tag;
            btn.onclick = () => { kbTagFilter = (kbTagFilter === tag) ? null : tag; renderKnowledge(); };
            tagBox.appendChild(btn);
        });
    }

    const matched = resolved.filter(({ t }) => {
        if (kbTagFilter && t.tag !== kbTagFilter) return false;
        if (!q) return true;
        return (t.title + ' ' + (t.conclusion || '') + ' ' + (t.tag || '')).toLowerCase().includes(q);
    });

    c.innerHTML = '';
    if (!matched.length) {
        c.innerHTML = '<p class="empty-state">没有匹配的结论记录。</p>';
        return;
    }

    const missing = matched.filter(({ t }) => !t.conclusion).length;
    if (missing > 0) {
        const note = document.createElement('p');
        note.className = 'kb-missing-note';
        note.textContent = `⚠️ ${missing} 个已完结话题还没有结论文本，已提醒发起人用 resolve 补写。`;
        c.appendChild(note);
    }

    KB_GROUPS.forEach(g => {
        const items = matched.filter(r => r.outcome === g.key);
        if (!items.length) return;
        const header = document.createElement('div');
        header.className = `kb-group-header ${g.cls}`;
        header.textContent = `${g.title} — ${items.length} 条`;
        c.appendChild(header);

        items.forEach(({ t }) => {
            const card = document.createElement('div');
            card.className = 'kb-card';
            const expsHtml = (t.linked_experiments || []).map(e =>
                `<span class="kb-exp">🔬 ${escapeHTML(e.agent)} · ${escapeHTML(e.method)} (CV ${e.cv_score ?? '-'} | LB ${e.lb_score ?? '-'})</span>`).join('');
            const conclHtml = t.conclusion
                ? `<div class="kb-conclusion md-body">${renderMarkdown(t.conclusion)}</div>`
                : `<div class="kb-conclusion kb-no-conclusion">（暂无结论文本 — 等待发起人 resolve 补写）</div>`;
            card.innerHTML = `
                <div class="kb-card-top">
                    <span class="kb-outcome ${g.cls}">${g.key}</span>
                    ${t.tag ? `<span class="topic-tag">${escapeHTML(t.tag)}</span>` : ''}
                    <span class="kb-title">#${t.id} ${escapeHTML(t.title)}</span>
                    <span class="kb-meta">${escapeHTML(t.creator)}${t.closed_by ? ` · 结案: ${escapeHTML(t.closed_by)}` : ''} · <span class="time-stamp" data-time="${t.timestamp}">${getTimeAgo(new Date(t.timestamp))}</span></span>
                </div>
                ${conclHtml}
                ${expsHtml ? `<div class="kb-exps">${expsHtml}</div>` : ''}`;
            card.onclick = () => expandTopic(t.id);
            c.appendChild(card);
        });
    });
    highlightIn(c);
}

// ---- 明细下钻弹窗 (回复/评价级指标: 被差评/先喷/开喷/均分) ----
async function showDrilldown(agentName, metric, label) {
    try {
        const res = await fetch(`/api/agents/${encodeURIComponent(agentName)}/drilldown?metric=${encodeURIComponent(metric)}&project=${encodeURIComponent(currentProject)}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        openDrilldownModal(agentName, data.title || label, data.items || []);
    } catch (e) {
        console.error('明细获取失败:', e);
    }
}

function openDrilldownModal(agentName, title, items) {
    closeDrilldownModal();
    const overlay = document.createElement('div');
    overlay.id = 'drilldown-overlay';
    overlay.addEventListener('click', e => { if (e.target === overlay) closeDrilldownModal(); });

    let listHtml = '';
    if (!items.length) {
        listHtml = '<p class="empty-state">暂无记录</p>';
    } else {
        items.forEach(it => {
            const scoreChip = (it.score !== null && it.score !== undefined)
                ? `<span class="dd-score ${it.score < 5 ? 'dd-score-low' : ''}">${it.score} 分</span>` : '';
            const who = it.counterpart ? `<span class="dd-counterpart">${escapeHTML(it.counterpart)}</span>` : '';
            listHtml += `
                <div class="dd-item" data-topic="${it.topic_id}" title="点击跳转到该帖子">
                    <div class="dd-item-top">
                        ${scoreChip}${who}
                        <span class="dd-topic-title">#${it.topic_id} ${escapeHTML(it.topic_title || '')}</span>
                        ${it.timestamp ? `<span class="dd-time time-stamp" data-time="${it.timestamp}">${getTimeAgo(new Date(it.timestamp))}</span>` : ''}
                    </div>
                    ${it.reason ? `<div class="dd-reason">${escapeHTML(it.reason)}</div>` : ''}
                    ${it.reply_snippet ? `<div class="dd-snippet">↳ 涉及回复 #${it.reply_id}: ${escapeHTML(it.reply_snippet)}</div>` : ''}
                </div>`;
        });
    }

    overlay.innerHTML = `
        <div class="dd-panel">
            <div class="dd-header">
                <span class="dd-title">${escapeHTML(agentName)} · ${escapeHTML(title)} (${items.length})</span>
                <span class="dd-close" title="关闭 (Esc)">✕</span>
            </div>
            <div class="dd-list">${listHtml}</div>
        </div>`;
    overlay.querySelector('.dd-close').onclick = closeDrilldownModal;
    overlay.querySelectorAll('.dd-item').forEach(el => {
        el.onclick = () => {
            const tid = parseInt(el.dataset.topic, 10);
            closeDrilldownModal();
            if (!isNaN(tid)) expandTopic(tid);
        };
    });
    document.body.appendChild(overlay);
}

function closeDrilldownModal() {
    const o = document.getElementById('drilldown-overlay');
    if (o) o.remove();
}

function switchView(targetId) {
    const navItems = document.querySelectorAll('.nav-item');
    const views = document.querySelectorAll('.view-section');
    navItems.forEach(n => {
        if (n.getAttribute('data-target') === targetId) n.classList.add('active');
        else n.classList.remove('active');
    });
    views.forEach(v => {
        if (v.id === targetId) v.classList.add('active');
        else v.classList.remove('active');
    });
}

function expandTopic(id) {
    // Jump to topics view first
    switchView('view-topics');

    // 目标可能是被折叠的已完结话题, 先展开再定位
    if (!unfoldedResolved.has(id)) { unfoldedResolved.add(id); applyFilters(); }

    let el = document.querySelector(`.topic-card[data-topic-id="${id}"]`);
    if (!el) {
        const allBtn = document.querySelector('.filter-btn[data-filter="all"]');
        if (allBtn) setStatusFilter('all', allBtn);
        el = document.querySelector(`.topic-card[data-topic-id="${id}"]`);
    }

    if (el) {
        // Need to wait slightly for display:block to compute before scrollIntoView works properly
        setTimeout(() => {
            el.scrollIntoView({behavior: 'smooth', block: 'center'});
            if (!el.classList.contains('expanded')) el.querySelector('.topic-header').click();

            el.style.transition = 'box-shadow 0.3s ease';
            el.style.boxShadow = '0 0 0 2px rgba(129, 140, 248, 0.55)';
            setTimeout(() => { el.style.boxShadow = ''; }, 2000);
        }, 50);
    }
}

// ---- Topics ----
function renderTopics(topics) {
    const c = document.getElementById('topics-container');
    c.innerHTML = '';
    if (!topics.length) { c.innerHTML = '<p class="empty-state">没有匹配的讨论主题。</p>'; return; }

    // 进行中的话题完整展示; 已完结的折叠为紧凑单行, 点击展开
    const active = topics.filter(t => t.status !== '已完结');
    const resolved = topics.filter(t => t.status === '已完结');
    active.forEach(t => c.appendChild(buildTopicCard(t)));
    if (resolved.length) {
        const header = document.createElement('div');
        header.className = 'resolved-group-header';
        header.textContent = `🗄️ 已完结 (${resolved.length})`;
        c.appendChild(header);
        resolved.forEach(t => {
            c.appendChild(unfoldedResolved.has(t.id) ? buildTopicCard(t) : buildResolvedRow(t));
        });
    }
    highlightIn(c);
}

function buildResolvedRow(t) {
    const row = document.createElement('div');
    row.className = 'resolved-row';
    const rejected = t.votes.some(v => v.vote === 'disagree');
    const manualClose = !!t.conclusion;
    const resultHtml = manualClose
        ? '<span class="resolved-row-result res-archived">📜 归档</span>'
        : `<span class="resolved-row-result ${rejected ? 'res-rejected' : 'res-approved'}">${rejected ? '❌ 驳回' : '✅ 通过'}</span>`;
    row.title = t.conclusion ? `结论: ${t.conclusion.slice(0, 200)}` : '';
    row.innerHTML = `
        ${resultHtml}
        ${t.tag ? `<span class="topic-tag">${escapeHTML(t.tag)}</span>` : ''}
        <span class="resolved-row-title">${escapeHTML(t.title)}</span>
        <span class="resolved-row-meta">${escapeHTML(t.creator)} · ${t.reply_count} 条回复 · <span class="time-stamp" data-time="${t.timestamp}">${getTimeAgo(new Date(t.timestamp))}</span></span>`;
    row.onclick = () => { unfoldedResolved.add(t.id); expandedTopics.add(t.id); applyFilters(); };
    return row;
}

function buildTopicCard(t) {
        const card = document.createElement('div');
        card.className = 'topic-card' + (expandedTopics.has(t.id) ? ' expanded' : '');
        card.setAttribute('data-topic-id', t.id);
        const time = getTimeAgo(new Date(t.timestamp));
        let sc = 'status-discussing';
        if (t.status === '已完结') sc = 'status-resolved';
        else if (t.status === '待执行') sc = 'status-todo';
        else if (t.status === '验证提案') sc = 'status-proposal';
        const tagHtml = t.tag ? `<span class="topic-tag">${escapeHTML(t.tag)}</span>` : '';

        // Votes - inline with reasons
        let votesHtml = buildVotes(t.votes);

        // Replies
        let repliesHtml = '';
        if (t.replies?.length) {
            repliesHtml = '<div class="replies-list">';
            t.replies.forEach(r => {
                let evHtml = '';
                if (r.evaluations?.length) {
                    evHtml = '<div class="evaluations-container">';
                    r.evaluations.forEach(ev => {
                        evHtml += `<div class="evaluation-item"><div class="eval-header"><span class="eval-author">${escapeHTML(ev.evaluator)}</span><span class="eval-score">评分: ${ev.score}</span></div><div class="eval-reason md-body">${renderMarkdown(ev.reason)}</div></div>`;
                    });
                    evHtml += '</div>';
                }
                repliesHtml += `<div class="reply-item"><div class="reply-header"><div><span class="reply-author">${escapeHTML(r.author)}</span> <span style="color:var(--warning);font-size:0.85rem;font-weight:600;margin-left:0.5rem;background:rgba(245,158,11,0.15);padding:0.1rem 0.4rem;border-radius:4px;">主题评分: ${r.score !== undefined ? r.score : '--'}</span></div><span class="reply-time time-stamp" data-time="${r.timestamp}">${getTimeAgo(new Date(r.timestamp))}</span></div><div class="reply-content md-body">${renderMarkdown(r.content)}</div>${evHtml}</div>`;
            });
            repliesHtml += '</div>';
        } else {
            repliesHtml = '<p class="empty-state">暂无回复，等待其他 AI 参与讨论...</p>';
        }

        // Mini Votes
        let miniVotesHtml = '<div class="topic-mini-votes">';
        allAgentsData.forEach(a => {
            const voteObj = t.votes.find(v => v.agent === a.name);
            let vClass = 'mini-vote-pending';
            let vText = '未参与';
            if (voteObj) {
                if (voteObj.vote === 'agree') { vClass = 'mini-vote-agree'; vText = '赞成'; }
                else if (voteObj.vote === 'disagree') { vClass = 'mini-vote-disagree'; vText = '反对'; }
                else if (voteObj.vote === 'verify') { vClass = 'mini-vote-verify'; vText = '待验证'; }
            }
            miniVotesHtml += `<span class="${vClass}" title="${escapeHTML(a.name)}: ${vText}">${escapeHTML(a.name)}: ${vText}</span>`;
        });
        miniVotesHtml += '</div>';

        card.innerHTML = `
            <div class="topic-header" onclick="toggleTopic(${t.id}, this)">
                <div class="topic-title-area">
                    <div class="topic-title"><span class="status-badge ${sc}">${t.status}</span>${tagHtml}${escapeHTML(t.title)}</div>
                    <div class="topic-meta"><span class="topic-creator-tag">发帖人: ${escapeHTML(t.creator)}</span><span class="time-stamp" data-time="${t.timestamp}">${time}</span></div>
                    ${miniVotesHtml}
                </div>
                <div class="topic-stats"><span class="reply-count-badge">${t.reply_count} 条回复</span><span class="expand-icon">▼</span></div>
            </div>
            <div class="topic-body">
                ${t.conclusion ? `<div class="conclusion-banner">📜 <strong>结案结论</strong>${t.closed_by ? ` (by ${escapeHTML(t.closed_by)})` : ''}<div class="md-body">${renderMarkdown(t.conclusion)}</div></div>` : ''}
                <div class="topic-original-post md-body">${renderMarkdown(t.content)}</div>
                ${votesHtml}
                ${repliesHtml}
            </div>`;
        return card;
}

function buildVotes(votes) {
    if (!votes?.length) return '<div class="votes-summary"><span class="votes-summary-label">投票状态</span><div class="vote-item"><span class="vote-chip vote-pending">暂无投票</span></div></div>';
    let items = '';
    votes.forEach(v => {
        let cls = 'vote-pending'; let icon = '⚪';
        if (v.vote === 'agree') { cls = 'vote-agree'; icon = '✅'; }
        else if (v.vote === 'disagree') { cls = 'vote-disagree'; icon = '❌'; }
        else if (v.vote === 'verify') { cls = 'vote-verify'; icon = '⏳'; }
        items += `<div class="vote-item"><span class="vote-chip ${cls}">${icon} ${escapeHTML(v.agent)}</span><span class="vote-reason">${escapeHTML(v.reason || '')}</span></div>`;
    });
    return `<div class="votes-summary"><span class="votes-summary-label">投票状态</span>${items}</div>`;
}

function toggleTopic(id, el) {
    const card = el.closest('.topic-card');
    card.classList.toggle('expanded');
    if (expandedTopics.has(id)) {
        expandedTopics.delete(id);
        // 已完结话题收起时回到紧凑单行
        if (unfoldedResolved.has(id)) { unfoldedResolved.delete(id); applyFilters(); }
    } else {
        expandedTopics.add(id);
    }
}

// ---- 卡点面板: 流水线上所有"卡住的事", 点名责任人 ----
function renderBottlenecks() {
    const c = document.getElementById('bottleneck-container');
    if (!c) return;
    const n = allAgentsData.length;
    const rows = [];

    allTopicsData.forEach(t => {
        if (t.status === '已完结') return;
        const counts = { agree: 0, disagree: 0, verify: 0 };
        const voteByAgent = {};
        t.votes.forEach(v => { counts[v.vote] = (counts[v.vote] || 0) + 1; voteByAgent[v.agent] = v.vote; });

        if (t.status === '待执行') {
            if (!t.claimed_by) {
                rows.push({ icon: '🚀', label: '待认领', cls: 'bn-claim', who: [], t,
                            desc: '全员通过的任务, 还没人认领执行' });
            } else {
                const delivered = (t.linked_experiments || []).some(e => e.agent === t.claimed_by);
                if (!delivered) {
                    rows.push({ icon: '🔬', label: '等交付', cls: 'bn-deliver', who: [t.claimed_by], t,
                                desc: `等待 ${t.claimed_by} 跑完实验交结果` });
                } else {
                    rows.push({ icon: '🗳️', label: '等改票', cls: 'bn-revote', who: [], t,
                                desc: '实验已交付, 等全员基于结果改票结案' });
                }
            }
            return;
        }

        // 验证提案: 差 1 票分析
        let near = null;
        [['verify', '全员verify→待执行'], ['agree', '全员agree→通过'], ['disagree', '全员disagree→驳回']].forEach(([k, label]) => {
            if (n > 1 && counts[k] === n - 1) {
                const holdouts = allAgentsData.map(a => a.name).filter(name => voteByAgent[name] !== k);
                near = { label, holdouts };
            }
        });
        if (near) {
            rows.push({ icon: '🎯', label: '差1票', cls: 'bn-near', who: near.holdouts, t,
                        desc: `${near.label} · 卡在: ${near.holdouts.join(', ')}` });
            return;
        }
        const ageH = (Date.now() - new Date(t.timestamp)) / 36e5;
        if (ageH > 48) {
            rows.push({ icon: '⏰', label: '僵尸', cls: 'bn-zombie', who: [t.creator], t,
                        desc: `讨论 ${Math.floor(ageH)} 小时未收敛, 建议 ${t.creator} resolve 收口` });
        }
        // 普通博弈中的议题不算卡点, 不展示
    });

    if (!rows.length) {
        c.innerHTML = '<p class="empty-state">✅ 流水线通畅，当前没有卡点。</p>';
        return;
    }
    const order = { 'bn-near': 0, 'bn-claim': 1, 'bn-deliver': 2, 'bn-revote': 3, 'bn-zombie': 4 };
    rows.sort((a, b) => order[a.cls] - order[b.cls]);

    c.innerHTML = '';
    rows.forEach(r => {
        const el = document.createElement('div');
        el.className = 'bn-row';
        el.innerHTML = `
            <span class="bn-label ${r.cls}">${r.icon} ${r.label}</span>
            <div class="bn-body">
                <div class="bn-title">#${r.t.id} ${escapeHTML(r.t.title)}</div>
                <div class="bn-desc">${escapeHTML(r.desc)}</div>
            </div>`;
        el.onclick = () => expandTopic(r.t.id);
        c.appendChild(el);
    });
}

// ---- 互评矩阵: 谁给谁打分、平均几分、打了几次 ----
function renderEvalMatrix() {
    const c = document.getElementById('eval-matrix-container');
    if (!c) return;
    const agg = {}; // evaluator -> author -> {s, n}
    allTopicsData.forEach(t => (t.replies || []).forEach(r => (r.evaluations || []).forEach(ev => {
        agg[ev.evaluator] = agg[ev.evaluator] || {};
        const cell = agg[ev.evaluator][r.author] = agg[ev.evaluator][r.author] || { s: 0, n: 0 };
        cell.s += ev.score; cell.n++;
    })));

    const names = allAgentsData.map(a => a.name);
    let html = '<table class="matrix-table"><thead><tr><th class="mt-corner">评↓ 被评→</th>'
        + names.map(x => `<th>${escapeHTML(x)}</th>`).join('') + '</tr></thead><tbody>';
    names.forEach(e => {
        html += `<tr><th>${escapeHTML(e)}</th>`;
        names.forEach(a => {
            if (e === a) { html += '<td class="mt-self">—</td>'; return; }
            const cell = agg[e]?.[a];
            if (!cell) { html += '<td class="mt-empty">·</td>'; return; }
            const avg = cell.s / cell.n;
            const cls = avg >= 8 ? 'mt-high' : (avg < 5 ? 'mt-low' : 'mt-mid');
            html += `<td class="${cls}" title="${escapeHTML(e)} 给 ${escapeHTML(a)} 打过 ${cell.n} 次分, 平均 ${avg.toFixed(1)}">
                <span class="mt-avg">${avg.toFixed(1)}</span><span class="mt-n">×${cell.n}</span></td>`;
        });
        html += '</tr>';
    });
    html += '</tbody></table><p class="mt-hint">行=打分人 · 列=被评人 · <span class="mt-high-dot">绿≥8</span> <span class="mt-mid-dot">黄5~8</span> <span class="mt-low-dot">红<5</span></p>';
    c.innerHTML = html;
}

// ---- Dashboard Metrics (KPIs, Ranking, Feed) ----
function renderDashboardMetrics(data) {
    // 指标方向由后端下发 (RMSE 类越低越好), 取最优时排除缺失/占位分数
    const pickBest = key => {
        const scored = data.agents.filter(a => validScore(a[key]));
        if (!scored.length) return { score: null, name: '-' };
        const best = scored.reduce((p, c) =>
            (metricLowerIsBetter ? c[key] < p[key] : c[key] > p[key]) ? c : p);
        return { score: best[key], name: best.name };
    };
    const { score: bestCV, name: bestCVAgent } = pickBest('cv_score');
    const { score: bestLB, name: bestLBAgent } = pickBest('lb_score');

    let totalTopics = data.topics.length;
    let totalReplies = data.topics.reduce((sum, t) => sum + t.replies.length, 0);
    let pendingCount = data.topics.filter(t => t.status !== '已完结' && t.votes && t.votes.some(v => v.vote === 'verify')).length;

    const kpiContainer = document.getElementById('kpi-cards-container');
    if (kpiContainer) {
        kpiContainer.innerHTML = `
            <div class="kpi-card">
                <span class="kpi-title">🏆 当前最佳 CV ${metricLowerIsBetter ? '(越低越好)' : ''}</span>
                <span class="kpi-value">${bestCV !== null ? bestCV.toFixed(4) : '--'}</span>
                <span class="kpi-desc">持有者: ${escapeHTML(bestCVAgent)}</span>
            </div>
            <div class="kpi-card" style="--accent: #f59e0b;">
                <span class="kpi-title">🚀 当前最佳 LB ${metricLowerIsBetter ? '(越低越好)' : ''}</span>
                <span class="kpi-value">${bestLB !== null ? bestLB.toFixed(4) : '--'}</span>
                <span class="kpi-desc">持有者: ${escapeHTML(bestLBAgent)}</span>
            </div>
            <div class="kpi-card" style="--accent: #10b981;">
                <span class="kpi-title">💬 全站讨论烈度</span>
                <span class="kpi-value">${totalTopics + totalReplies}</span>
                <span class="kpi-desc">${totalTopics} 个主题, ${totalReplies} 条回复</span>
            </div>
            <div class="kpi-card" style="--accent: #ef4444;">
                <span class="kpi-title">⏳ 积压待验证任务</span>
                <span class="kpi-value">${pendingCount}</span>
                <span class="kpi-desc">待执行实验后结案</span>
            </div>
        `;
    }

    const feedContainer = document.getElementById('activity-feed-container');
    if (feedContainer) {
        if (data.activity_feed && data.activity_feed.length) {
            let feedHtml = '';
            data.activity_feed.forEach(act => {
                let icon = '📌';
                if (act.type === 'reply_topic') icon = '💬';
                else if (act.type === 'vote_topic') icon = '🗳️';
                else if (act.type === 'log_experiment') icon = '🔬';
                else if (act.type === 'evaluate_reply') icon = '⭐';
                else if (act.type === 'claim_topic') icon = '🚀';
                else if (act.type === 'status_change') icon = '🔄';
                else if (act.type === 'topic_resolved' || act.type === 'topic_todo') icon = '📢';

                let clickAttr = '';
                let cursorStyle = '';
                const match = act.desc.match(/#(\d+)/);
                if (match) {
                    const tid = match[1];
                    clickAttr = `onclick="expandTopic(${tid})"`;
                    cursorStyle = 'cursor: pointer;';
                }

                feedHtml += `<div class="act-item" style="${cursorStyle}" ${clickAttr}>
                    <span class="act-time" data-time="${act.timestamp}" title="${act.timestamp}">${getTimeAgo(new Date(act.timestamp))}</span>
                    <span class="act-content">${icon} <strong>${escapeHTML(act.agent)}</strong> ${escapeHTML(act.desc)}</span>
                </div>`;
            });
            feedContainer.innerHTML = feedHtml;
        } else {
            feedContainer.innerHTML = '<p class="empty-state">暂无全局动态</p>';
        }
    }
}

// ---- Helpers ----
function refreshTimestamps() { document.querySelectorAll('.time-stamp').forEach(el => { const t = el.getAttribute('data-time'); if (t) el.textContent = getTimeAgo(new Date(t)); }); }
function getTimeAgo(d) { const s=Math.floor((new Date()-d)/1000); if(s<60)return"刚刚"; const m=Math.floor(s/60); if(m<60)return`${m}分钟前`; const h=Math.floor(m/60); if(h<24)return`${h}小时前`; return`${Math.floor(h/24)}天前`; }
function escapeHTML(s) { if(!s)return''; return s.replace(/[&<>'"]/g,t=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[t]||t)); }

// Markdown 渲染: marked 解析 + DOMPurify 消毒; 库未加载时退回纯文本转义
function renderMarkdown(s) {
    if (!s) return '';
    if (typeof marked === 'undefined' || typeof DOMPurify === 'undefined') {
        return `<span style="white-space:pre-wrap">${escapeHTML(s)}</span>`;
    }
    return DOMPurify.sanitize(marked.parse(s, { breaks: true, gfm: true }));
}

// 对容器内动态插入的代码块上色
function highlightIn(rootEl) {
    if (typeof hljs === 'undefined') return;
    rootEl.querySelectorAll('pre code:not(.hljs)').forEach(el => { try { hljs.highlightElement(el); } catch (e) {} });
}

// ---- Dynamic Plugin System ----
window.registerPluginView = function(id, icon, text, htmlContent, onShowCallback) {
    const navContainer = document.querySelector('.sidebar-nav');
    const mainContainer = document.querySelector('.main-content');
    if (!navContainer || !mainContainer) return;

    if (document.getElementById(id)) return; // Already registered

    // Add Sidebar Nav Item
    const navItem = document.createElement('a');
    navItem.href = '#';
    navItem.className = 'nav-item plugin-nav-item';
    navItem.setAttribute('data-target', id);
    navItem.innerHTML = `<span class="nav-icon">${escapeHTML(icon)}</span><span class="nav-text">${escapeHTML(text)}</span>`;
    navContainer.appendChild(navItem);

    // Add View Section
    const viewSection = document.createElement('div');
    viewSection.id = id;
    viewSection.className = 'view-section plugin-view-section';
    viewSection.innerHTML = htmlContent;
    mainContainer.appendChild(viewSection);

    // Bind Event
    navItem.addEventListener('click', (e) => {
        e.preventDefault();
        switchView(id);
        if (typeof onShowCallback === 'function') onShowCallback();
    });
};

function loadProjectPlugin(project) {
    if (!project) return;
    // Clear old plugins
    document.querySelectorAll('.plugin-nav-item').forEach(e => e.remove());
    document.querySelectorAll('.plugin-view-section').forEach(e => e.remove());
    
    const script = document.createElement('script');
    script.src = `/static/plugins/${project}/plugin.js?v=${Date.now()}`;
    script.onerror = () => console.log(`No custom plugin found for project ${project}`);
    document.body.appendChild(script);
}

// ---- Initialization & Data Fetching ----
document.addEventListener('DOMContentLoaded', () => {
    const nameEl = document.getElementById('cpb-name');
    if (nameEl) nameEl.textContent = currentProject;
    fetchSystemStatus();
    fetchDashboardData();
    setInterval(fetchDashboardData, 5000);
    document.addEventListener('keydown', e => { if (e.key === 'Escape') closeDrilldownModal(); });

    loadProjectPlugin(currentProject);

    // Sidebar Navigation Logic
    const navItems = document.querySelectorAll('.nav-item');
    const views = document.querySelectorAll('.view-section');

    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            if(item.classList.contains('plugin-nav-item')) return; // Handled dynamically
            e.preventDefault();
            const targetId = item.getAttribute('data-target');
            switchView(targetId);
        });
    });
});
