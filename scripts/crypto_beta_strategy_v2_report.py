"""Separate analytical Markdown and offline visual HTML research artifacts."""
from __future__ import annotations
import base64,html,json,re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/crypto/market_beta_strategy_v2'
NAMES={'trend24':'24h 快趋势','trend72':'72h 中趋势','trend168':'168h 慢趋势','trend336':'336h 长趋势','trend_ensemble':'三速趋势组合','breakout':'通道位置趋势','range_only':'区间反转单专家','down_rebound':'跌势/反弹切换','selected_trend':'季度本币择速','selected_breakout':'季度本币通道选择','regime_gate':'经济状态门控','gate_no_range':'门控移除反转','adaptive14':'状态专家·14d 记忆','adaptive42':'状态专家·42d 记忆','adaptive126':'状态专家·126d 记忆','filtered_moe':'滤波隐状态专家','kalman':'Kalman 潜在漂移','trend_ensemble_long':'趋势组合·多头','regime_gate_long':'经济门控·多头','expanding_tree':'扩展多期限树','rolling_tree':'滚动多期限树','action_tcn':'直接动作 TCN','vol_long':'波动控制多头','hold':'固定数量持有'}
GROUPS=[('01｜趋势速度',['trend24','trend72','trend168']),('02｜长趋势与期限适配',['trend336','trend_ensemble','selected_trend']),('03｜路径与潜在漂移',['breakout','selected_breakout','kalman']),('04｜交易机制切换',['range_only','down_rebound','regime_gate']),('05｜条件记忆长度',['adaptive14','adaptive42','adaptive126']),('06｜统计关系更新',['expanding_tree','rolling_tree','action_tcn']),('07｜隐状态的增量',['filtered_moe','regime_gate','gate_no_range']),('08｜多头参考',['hold','vol_long','trend_ensemble_long']),('09｜门控多空对照',['regime_gate','regime_gate_long','vol_long'])]
FOCUS=['trend168','breakout','selected_trend','regime_gate','adaptive42','filtered_moe','rolling_tree','action_tcn']
COLORS=['#0f766e','#f59e0b','#6366f1']
REGIME_NAMES={'up_persistent':'上行持续','down_persistent':'下行持续','panic_continuation':'加速下跌','rebound':'跌后反弹','range':'低持续区间','transition':'过渡'}

def table(f):return f.to_markdown(index=False)
def pct(v):return f'{v:+.2%}'
def embed(path):return 'data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()
def draw(scores,portfolio,contracts,legs,attrib,episodes):
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'#f8fafc','axes.edgecolor':'#cbd5e1','font.size':10,'grid.alpha':.18})
    d=portfolio[portfolio.fee_bp==2].pivot(index='day',columns='policy',values='net_return');wealth=(1+d).cumprod()
    for j,(title,policies) in enumerate(GROUPS):
        fig,ax=plt.subplots(figsize=(10,5))
        for i,p in enumerate(policies):ax.plot(wealth.index,wealth[p],color=COLORS[i],label=NAMES[p],lw=1.8)
        ax.axvspan(pd.Timestamp('2026-01-01',tz='UTC'),wealth.index[-1],color='#e2e8f0',alpha=.65)
        ax.set(title=title,ylabel='12 个独立子账户财富（对数轴）',yscale='log');ax.grid()
        ax.legend(loc='upper left',frameon=False,ncol=1);fig.tight_layout();fig.savefig(OUT/f'portfolio_{j+1:02}.png',dpi=150);plt.close(fig)
    for phase in ['all','reused_2026']:
        x=scores[(scores.symbol!='PORT12')&(scores.fee_bp==2)&(scores.phase==phase)].pivot(index='policy',columns='symbol',values='return_total').reindex(NAMES)
        fig,ax=plt.subplots(figsize=(13,12));im=ax.imshow(x*100,cmap='RdYlGn',vmin=-80,vmax=100,aspect='auto')
        for i in range(len(x)):
            for j in range(len(x.columns)):ax.text(j,i,f'{x.iloc[i,j]*100:+.0f}',ha='center',va='center',fontsize=8,color='#172554')
        ax.set_xticks(range(len(x.columns)),[s.replace('USDT','') for s in x.columns],rotation=45,ha='right');ax.set_yticks(range(len(x)),[NAMES[p] for p in x.index])
        ax.set(title='所有策略 × 所有合约｜'+('2024–2026 全期' if phase=='all' else '2026 已复用诊断')+'，maker 2bp，数字为累计 %')
        fig.colorbar(im,ax=ax,label='累计收益 %（颜色截断，文字保留原值）',shrink=.5);fig.tight_layout();fig.savefig(OUT/f'heatmap_{phase}.png',dpi=150);plt.close(fig)
    for s in contracts.symbol.unique():
        d=contracts[(contracts.symbol==s)&(contracts.fee_bp==2)].pivot(index='day',columns='policy',values='net_return');w=(1+d).cumprod()
        fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True)
        for ax,policies,title in [(axes[0],['trend168','selected_trend','regime_gate'],'趋势与状态机制'),(axes[1],['adaptive42','rolling_tree','action_tcn'],'适应与直接动作')]:
            for i,p in enumerate(policies):ax.plot(w.index,w[p],label=NAMES[p],color=COLORS[i],lw=1.6)
            ax.axvspan(pd.Timestamp('2026-01-01',tz='UTC'),w.index[-1],color='#e2e8f0',alpha=.6);ax.set(title=title,ylabel='财富，对数轴',yscale='log');ax.legend(frameon=False,ncol=3,fontsize=9);ax.grid()
        fig.suptitle(s+'｜每张子图最多三条曲线 · maker 2bp · 价格减手续费',fontweight='bold',fontsize=14);fig.tight_layout();fig.savefig(OUT/f'coin_{s}.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,6))
    for ax,phase,title in zip(axes,['all','reused_2026'],['全期净利润按方向分解','2026 净利润按方向分解']):
        x=legs[(legs.symbol=='PORT12')&(legs.phase==phase)].set_index('policy').reindex(FOCUS)
        y=np.arange(len(x));ax.barh(y-.17,x.long_net_contribution*100,height=.32,label='多头净贡献',color='#0f766e');ax.barh(y+.17,x.short_net_contribution*100,height=.32,label='空头净贡献',color='#f59e0b');ax.axvline(0,color='#94a3b8');ax.set_yticks(y,[NAMES[p] for p in x.index]);ax.set(title=title,xlabel='相对该段初始组合资金的利润 %');ax.legend(frameon=False);ax.grid(axis='x')
    fig.tight_layout();fig.savefig(OUT/'side_contributions.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,6))
    x=attrib[(attrib.policy=='selected_trend')&(attrib.phase=='all')].set_index('symbol').sort_index();y=np.arange(len(x))
    for offset,col,label,color in [(-.25,'average_exposure_sum','平均市场暴露','#94a3b8'),(0,'timing_covariance_sum','实际择时协方差','#0f766e'),(.25,'residual_sum','本币剩余价格项','#f59e0b')]:
        ax.barh(y+offset,x[col]*100,height=.24,label=label,color=color)
    ax.set_yticks(y,[s.replace('USDT','') for s in x.index]);ax.axvline(0,color='#64748b');ax.set(title='季度本币择速｜共同风险与择时分解（非复利收益）',xlabel='5m 算术价格收益贡献加总 %');ax.legend(frameon=False);ax.grid(axis='x');fig.tight_layout();fig.savefig(OUT/'timing_components.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,5.5))
    data=[];positions=[];labels=[]
    for i,p in enumerate(FOCUS):
        for sign,off in [(1,-.17),(-1,.17)]:
            g=episodes[(episodes.policy==p)&(episodes.sign==sign)&(~episodes.right_censored)]
            data.append(g.hours.to_numpy());positions.append(i+off)
        labels.append(NAMES[p])
    boxes=ax.boxplot(data,positions=positions,widths=.27,showfliers=False,patch_artist=True,manage_ticks=False)
    for j,b in enumerate(boxes['boxes']):b.set_facecolor('#0f766e' if j%2==0 else '#f59e0b');b.set_alpha(.6)
    ax.set(yscale='log',title='已完成方向持仓段｜绿=多头，橙=空头；末段删失单列',ylabel='持续小时（对数轴）');ax.set_xticks(range(len(FOCUS)),labels,rotation=15);ax.axhline(12,color='#94a3b8',ls=':');ax.axhline(264,color='#94a3b8',ls=':');ax.grid(axis='y');fig.tight_layout();fig.savefig(OUT/'holding_duration.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,6))
    for ax,phase,title in zip(axes,['all','reused_2026'],['全期费用敏感性','2026 费用敏感性']):
        x=scores[(scores.symbol=='PORT12')&(scores.phase==phase)].pivot(index='policy',columns='fee_bp',values='return_total').reindex(FOCUS)
        y=np.arange(len(x));ax.barh(y-.17,x[2]*100,height=.32,label='maker 2bp',color='#0f766e');ax.barh(y+.17,x[5]*100,height=.32,label='taker 5bp',color='#6366f1');ax.set_yticks(y,[NAMES[p] for p in x.index]);ax.axvline(0,color='#94a3b8');ax.set(title=title,xlabel='累计收益 %');ax.legend(frameon=False);ax.grid(axis='x')
    fig.tight_layout();fig.savefig(OUT/'cost_comparison.png',dpi=150);plt.close(fig)

def main():
    scores=pd.read_csv(OUT/'strategy_scores.csv');portfolio=pd.read_csv(OUT/'portfolio_daily.csv',parse_dates=['day']);contracts=pd.read_csv(OUT/'contract_daily.csv.gz',parse_dates=['day'])
    legs=pd.read_csv(OUT/'side_contribution.csv');attrib=pd.read_csv(OUT/'timing_attribution.csv');episodes=pd.read_csv(OUT/'episodes.csv');fits=pd.read_csv(OUT/'fit_audit.csv');choices=pd.read_csv(OUT/'parameter_choices.csv')
    opportunity=pd.read_csv(OUT/'state_future_opportunities.csv');transfer=pd.read_csv(OUT/'signal_transfer_cards.csv');manifest=json.loads((OUT/'run_manifest.json').read_text())
    draw(scores,portfolio,contracts,legs,attrib,episodes)
    ps=scores[(scores.symbol=='PORT12')&(scores.fee_bp==2)]
    focus=ps[(ps.phase=='all')&ps.policy.isin(FOCUS)].set_index('policy').reindex(FOCUS)
    all_returns=ps[ps.phase=='all'].set_index('policy').return_total
    reused_returns=ps[ps.phase=='reused_2026'].set_index('policy').return_total
    summary=[]
    for p in NAMES:
        a=ps[(ps.policy==p)&(ps.phase=='all')].iloc[0];b=ps[(ps.policy==p)&(ps.phase=='reused_2026')].iloc[0]
        l=legs[(legs.symbol=='PORT12')&(legs.policy==p)&(legs.phase=='all')].iloc[0]
        count=scores[(scores.symbol!='PORT12')&(scores.fee_bp==2)&(scores.policy==p)&(scores.phase=='all')].return_total.gt(0).sum()
        summary.append({'策略':NAMES[p],'全期收益':pct(a.return_total),'2026复用':pct(b.return_total),'Sharpe':f'{a.sharpe:.2f}','最大回撤':pct(a.max_dd),'盈利合约':f'{count}/12','多头净贡献':pct(l.long_net_contribution),'空头净贡献':pct(l.short_net_contribution)})
    summary=pd.DataFrame(summary)
    latest=choices[choices.fold==choices.fold.max()].pivot(index='symbol',columns='kind',values='selected').reset_index()
    text=[]
    def add(s):text.append(s)
    add('# 市场 Beta 策略深研 V2｜机制、空头与适应性\n\n2026-10-04 · 独立方案 A 分支 · 12 个 USDT 永续 · maker 2bp 主情景、taker 5bp 对照\n\n本 Markdown 负责核心分析、模型公式、论文理解与取舍；可视化与可交互结果入口见同目录 `MARKET_BETA_STRATEGY_V2.html`。两份文档分工明确，不将图片和完整正文重复堆叠。')
    add('## 01　先给出研究判断\n\n本轮由策略机制主导，已对全部 12 合约运行 24 种规则、状态、适应与监督方案。重点不是再把数据验一遍，而是考察同一市场风险在不同路径下应该持有、反转还是等待。连续小时信号、4h 复核和固定数量账本复用上一轮平台；初始等资金的 12 个独立子账户组成实际组合。')
    add(f'固定机制参照中，168h 趋势全期 {pct(all_returns.trend168)}、2026 {pct(reused_returns.trend168)}；通道位置趋势全期 {pct(all_returns.breakout)}、2026 {pct(reused_returns.breakout)}。季度本币择速全期 {pct(all_returns.selected_trend)}、2026 {pct(reused_returns.selected_trend)}；42 日条件专家全期 {pct(all_returns.adaptive42)}、2026 {pct(reused_returns.adaptive42)}；滚动多期限树全期 {pct(all_returns.rolling_tree)}、2026 {pct(reused_returns.rolling_tree)}。机制对照共同展示，全部结果、失败策略与币种都留在表中。')
    add('2026 已被项目反复读取。本文的“前向”表示每折训练、选择和适应只用当时已知信息，是算法的时间逻辑；它不能清除研究者看过历史的事实，因此本轮仍是历史开发与复用诊断，不称未触碰 OOS。')
    add('## 02　策略产品与研究问题\n\n持仓范围接受十几小时到十多天。趋势窗口为 24/72/168/336h，监督持有目标 24/72/240h，实际方向 episode 由信号延续和反转产生；窗口、标签期限和最终持仓长度不是同一个概念。多空目标名义上限均为 1，风险缩放为 min(1,45%/(已知小时 RMS×√8760))，不是把组合实际波动强制校准到 45%。\n\n需要回答：同一合约为何应该现在承担正或负风险？下跌延续与反弹如何区分？区间应转为反转策略，还是降低参与？历史收益关系的记忆需要多久？12 个币都赚钱是否只是共同市场行情？')
    add('## 03　原始论文如何改变策略理解\n\n[时间序列动量研究](https://www.aqr.com/Insights/Research/Journal-Article/Time-Series-Momentum)支持独立本币趋势对照。[趋势凸性论文](https://arxiv.org/html/1607.02410)提醒收益依赖长短尺度方差与持续性，并受观测期限影响。[Momentum crashes](https://spinup-000d1a-wp-offload-media.s3.amazonaws.com/faculty/wp-content/uploads/sites/3/2019/09/Mom_crashes_JFE_final.pdf)给出跌后急反弹的重要警示。[状态配置研究](https://business.columbia.edu/sites/default/files-efs/pubfiles/1959/inquire.pdf)强调条件配置和前向状态概率。[直接仓位网络](https://arxiv.org/html/1904.04912)启发动作目标与换手正则。[动态在线学习](https://arxiv.org/html/1810.10815)启发多记忆专家。每项在本项目的迁移都是待检验假说，市场和期限不同；详细章节阅读与摘要/全文访问边界见 `docs/MARKET_BETA_STRATEGY_V2_LITERATURE.md`。')
    add('## 04　风险 Beta 的核心：规模、方向与时间是三件事\n\n对合约 i，r_i≈β_i r_m+ε_i，其中 m 固定为 BTC/ETH 等权市场，β_i 用当时已完成小时收益的 EW 协方差/方差估计（半衰期 360h，至少 720h）。策略共同市场暴露为 B_i=w_iβ_i。名义仓位 w_i、回归 beta β_i、实际组合风险都不同；高 beta 合约的小仓也可能承担很强的共同风险。\n\n“高波状态”描述风险大小；方向持续、市场广度与反弹路径描述收益机会。高波持续下跌可以支持负 Beta，高波急反弹则可能要求退出短腿；低波持续下跌也可做空，不能被低波标签挡掉。共同相关提高时，12 个合约并不是 12 个独立风险。监督模型共享本币尺度、市场趋势、广度、beta 与相关字段，而规则门控以本币路径为主，这是两类不同策略。\n\n能量比 E[(Σ72h r)^2]/(72 E[r²]) 用来描述大尺度变动相对短期噪声的大小；它未去均值，包含市场漂移，不等于标准中心化 variance ratio。本轮仅作为条件输入，不能单独解释为可预测自相关。')
    add('## 05　“择时协方差”究竟指什么\n\n```text\nmean(B_t × r_m,t→t+1)\n= mean(B_t) × mean(r_m,t→t+1) + Cov_population(B_t, r_m,t→t+1)\n策略价格项 = 共同市场项 + 本币剩余项\n策略净项 = 价格项 − 手续费 [−资金费率，另列已有数据敏感性]\n```\n\n第一项来自平均暴露乘市场漂移，第二项来自暴露高低/正负与随后收益的配合。平均净空遇上下跌可以仅赚漂移项；平均暴露接近零的趋势也可以赚择时项。例：两个时期市场先 +2%、后 −2%，事前仓位分别 +1/−1，则平均暴露和平均市场收益都为零，但平均价格 PnL 为 +2%，全部来自协方差；仓位反过来则为 −2%。这只是恒等式示例，不是可预知未来的策略。\n\n区别于资产协方差 Cov(r_i,r_m)，也区别于跟过去涨跌的 Cov(B_t,past r_m)。本轮用实际 5m 数量暴露×事前小时 beta 计算，恒等式精确；加总的是算术收益，不能读作复利收益。另存目标市场暴露与下一小时收益的协方差，辅助识别权重自然漂移的机械部分。本币剩余项包含 beta 误差和估计偏差，不自动称 alpha。')
    at=attrib[(attrib.policy=='selected_trend')&(attrib.phase=='all')].copy()
    at['合约']=at.symbol;at['平均暴露项']=at.average_exposure_sum.map(pct);at['择时协方差项']=at.timing_covariance_sum.map(pct);at['剩余价格项']=at.residual_sum.map(pct)
    add(table(at[['合约','平均暴露项','择时协方差项','剩余价格项']])+'\n\n该表为季度本币择速的 5m 算术分解；独立逐币，不将加总冒充组合复利。')
    add('## 06　24 种方案的策略内核\n\n| 家族 | 实际实现 | 回答的策略问题 |\n|---|---|---|\n| 速度 | 24/72/168/336h；三速组合 | 能否捕捉不同形成速度的趋势 |\n| 本币选择 | 每季度从快/中/慢或 48/168/336h 通道选择 | 不同合约是否需要不同记忆 |\n| 通道与漂移 | 当前价格相对过去通道位置；Kalman 局部漂移 | 路径位置或潜在方向是否比简单过去收益更好 |\n| 分工 | 区间反转、跌势/反弹、经济门控、移除反转消融 | 不同状态是否应调用不同交易机制 |\n| 适应 | 状态条件在线专家，14/42/126 日遗忘 | 旧局部收益关系应保留多久 |\n| 隐状态 | 四维隐区概率的前向滤波专家混合 | 连续状态身份是否有额外动作价值 |\n| 监督 | 扩展/滚动 24/72/240h 浅树 | 条件收益关系与训练记忆怎样共同决定方向 |\n| 动作 | 因果卷积 TCN 直接连续仓位 | 直接学动作是否优于均值再映射 |\n| 参考 | 固定持有、波动多头、两个多头版本 | 多空增量是否真的来自短腿 |')
    add('趋势强度是 tanh(标准化过去收益/0.8)；长趋势组合有固定饱和尺度，避免弱趋势无限放大。通道策略是连续位置强度，不是只在越过 Donchian 边界时触发的纯突破系统。Kalman 用已知波动作为观测噪声，局部漂移按固定过程/观测噪声比例 0.003 递推。两者都有其假设，不是“更真实”的方向。\n\n季度选择仅用此前 365 日、已完全成熟的 1h 价格收益，126 日衰减，按均值/RMS 与 2bp 变化成本代理选一个速度。选择目标是低成本近似，不是训练内全量 5m 最优 Sharpe；所有选择表留存。')
    add('## 07　经济状态与专家怎样联动\n\n状态分为上行持续、下行持续、加速下跌、跌后反弹、低持续区间、过渡。反弹优先判别：仍在过去 168h 回撤中，但近 6h 转强、24h 恢复显著且加速度为正；随后判别高波延续下跌，再判别普通下跌/上涨及区间。全部是当前路径条件，没有在状态标签里偷放未来收益。\n\n上行/下行持续主要配置中慢趋势与通道；加速下跌增加快趋势和跌势专家；反弹增加快方向和反弹专家；低持续区间增加反转；过渡保留更慢参考与现金。七个专家为快、中、慢、通道、反转、跌势反弹和现金。经济先验可以错，尤其低持续性不保证均值回复；因此把反转单专家与移除反转消融直接列入比较。\n\n门控是策略类型选择，不只是波动缩放。短腿使用与长腿相同名义上限，但跌势/反弹逻辑刻意不完全镜像：下跌速度、空头挤压及反弹路径的经济结构不同。')
    add('## 08　从 2026 失效到可适应的关系学习\n\n旧模型可能发生输入尺度漂移、状态内条件收益关系漂移、速度错配或动作映射错配。一个滚动标准化器只能处理部分输入尺度，无法修复收益符号变了。当前作三种实际尝试：\n\n1. 条件在线专家：按已实现收益更新所在旧状态的专家表现，14/42/126 日 EW 记忆，每状态 168h 先验收缩，以经济先验混合指数权重。t+5m→t+1h5m 的收益在 t+2h 完成快照时才可更新。费用使用连续专家变化代理。\n2. 滤波专家：季度近 365 日拟合四成分高斯发射，经验转移加 sticky 对角先验，训练条件 24h 专家价值强收缩 720h，未来只递推 P(z_t|信息≤t)。GMM 发射+经验 Markov 不是完整联合 HMM EM；训练期滤波同样向前，标签成熟后才校准专家动作。隐编号每折重学，不宣称永久稳定原型。\n3. 监督记忆：相同树容量，扩展训练对照近 365 日+126 日衰减。12 币池化、无币名类别，使用本币标准化尺度；24/72/240h 归一化收益分别预测，固定 0.45/0.35/0.20 权重映射为连续方向。并非任意将三段收益相加作为金额。')
    add(f'相同容量的扩展树与滚动树：全期 {pct(all_returns.expanding_tree)} / {pct(all_returns.rolling_tree)}，2026 {pct(reused_returns.expanding_tree)} / {pct(reused_returns.rolling_tree)}。这才是本轮对“旧统计关系迁移”较直接的比较；不能把它们与旧 BTC/ETH、6bp、35% 风险预算的 V1 收益直接相减。滚动改善某段并不意味着解决所有非平稳关系。')
    drift=[]
    for policy in ['trend168','regime_gate','adaptive42','expanding_tree','rolling_tree','action_tcn']:
        for phase in ['development_2024_25','reused_2026']:
            c=transfer[(transfer.policy==policy)&(transfer.phase==phase)]
            drift.append({'策略':NAMES[policy],'阶段':phase,'12币平均前向24h IC':f'{c.signal_return24_ic.mean():+.4f}','IC为负的合约':f'{c.signal_return24_ic.lt(0).sum()}/12'})
    add('同一信号与随后 24h 价格收益的联系（只是描述，标签重叠，非独立显著性）：\n\n'+table(pd.DataFrame(drift)))
    add('监督失效不是简单由手续费解释：2026 的滚动树在 11/12 合约的信号—未来 24h 关联为负；直接动作网络在 10/12 为负。滚动窗口缩短后，有效独立行情更少、近期状态占比更集中，未必更能识别未来关系。树的训练目标是标准化多期限均值，仓位映射仍固定；预测标尺变化和持仓价值错配可能同时存在。这里支持“需要条件机会与转折机制”，不支持“再滚得更快即可恢复”。')
    add('## 09　直接动作 TCN：更复杂，但仍要知道它在学什么\n\n池化 12 币、约千参数的因果卷积网络；传入过去 42h 的 8 个 6h 快照，目前单层宽度 3 卷积取最后输出，**有效时序感受野是最后 12h**，早期快照未进入最终读出。每快照含最长 336h 的慢状态，所以不是仅看 12h 原始收益。这个容量限制作为明确设计边界，不包装成完整 42h 路径模型。\n\n输出 a=tanh(network)，实际目标 q=a×已知波动 cap。训练采用非重叠 4h 标签，每季度近一年，每轮固定 10 epoch。\n\n```text\nproxy_pnl = q_t R_4h − 0.0002 |q_t − q_previous4h|\nloss = −100 mean(proxy_pnl) +150 mean(proxy_pnl²)+0.003 mean(q²)\n```\n\n这是直接效用和变化惩罚的近似，未复制文献 Sharpe-LSTM。它没有把 4h 调仓死区和自然数量漂移放进反传图，因此训练效用与实际账本有差距；这个差距也是策略失败机制。下一步若值得保留，应尝试多尺度卷积/显式持有状态，而非盲目加深网络。')
    add('## 10　全部组合与全部合约结果\n\n主费用 maker 2bp；统一价格减手续费，不混用缺失资金费率。Sharpe 按日收益与 365 日年化，最大回撤按日净值；跨日/跨币共振不当作独立样本。盈利合约数是覆盖诊断，不是 12 次独立显著证据。\n\n'+table(summary))
    matrix=scores[(scores.symbol!='PORT12')&(scores.fee_bp==2)&(scores.phase=='all')].pivot(index='policy',columns='symbol',values='return_total').reindex(NAMES)
    matrix.index=[NAMES[p] for p in matrix.index];matrix.columns=[c.replace('USDT','') for c in matrix.columns]
    add('### 每个策略的 12 合约全期收益\n\n'+table(matrix.map(pct).reset_index(names='策略')))
    matrix26=scores[(scores.symbol!='PORT12')&(scores.fee_bp==2)&(scores.phase=='reused_2026')].pivot(index='policy',columns='symbol',values='return_total').reindex(NAMES);matrix26.index=[NAMES[p] for p in matrix26.index];matrix26.columns=[c.replace('USDT','') for c in matrix26.columns]
    add('### 每个策略的 12 合约 2026 复用结果\n\n'+table(matrix26.map(pct).reset_index(names='策略')))
    add('## 11　空头端：利润、时间和市场机会要分开\n\n下表将真实买卖数量变化的手续费归于多头或空头组件，反转两条腿各自付费，期末平仓亦计入。净贡献以该段初始资金为分母，两条腿之和精确等于该段复利总利润。它与先前逐期算术收益和是不同口径。组合各腿按段初真实子账户资金权重加总。')
    side=[]
    for p in FOCUS:
        for phase in ['all','reused_2026']:
            a=legs[(legs.symbol=='PORT12')&(legs.policy==p)&(legs.phase==phase)].iloc[0]
            b=ps[(ps.policy==p)&(ps.phase==phase)].iloc[0]
            side.append({'策略':NAMES[p],'阶段':phase,'多头净利润':pct(a.long_net_contribution),'空头净利润':pct(a.short_net_contribution),'多头时间':f'{b.long_time:.1%}','空头时间':f'{b.short_time:.1%}','短腿bp/名义小时':f'{a.short_net_bp_per_exposure_hour:.3f}'})
    add(table(pd.DataFrame(side)))
    short168=legs[(legs.symbol=='PORT12')&(legs.policy=='trend168')&(legs.phase=='reused_2026')].iloc[0]
    full168=legs[(legs.symbol=='PORT12')&(legs.policy=='trend168')&(legs.phase=='all')].iloc[0]
    add(f'本轮最关键的空头证据：168h 趋势在 2026 的短腿净贡献为 {pct(short168.short_net_contribution)}，多头为 {pct(short168.long_net_contribution)}，所以空头确实承担了该段主要利润；但全期短腿为 {pct(full168.short_net_contribution)}。这是市场阶段差异，不应追求各年各腿都赚同样的钱。经济门控全期短腿也为负，说明把下跌与反弹划出标签本身还没有提升短腿价值。')
    add('应该争取的是有效短腿：在可持续下跌早期进入，避免反弹时机械延续慢空仓，费后仍有正贡献。增加空头时间而降低每单位负 Beta 的质量并无意义；反弹转多也不能只凭曾经下跌，需要实际恢复路径。多头参考和双向策略的资金规模/漂移不同，因此空头净组件表比简单相减两条净值更直接。')
    duration=[]
    for p in FOCUS:
        for sign in [1,-1]:
            g=episodes[(episodes.policy==p)&(episodes.sign==sign)&(~episodes.right_censored)]
            duration.append({'策略':NAMES[p],'方向':'多' if sign==1 else '空','完成段数':len(g),'中位小时':f'{g.hours.median():.1f}','90分位小时':f'{g.hours.quantile(.9):.1f}','12–264h占比':f'{g.hours.between(12,264).mean():.1%}'})
    add(table(pd.DataFrame(duration))+'\n\n这是跨币已完成方向段的分布，币间同步段并不独立；末段右删失保存在原始表。不限死持仓期限，以策略方向有效为续持依据。')
    add('## 12　参数适配与状态转移的下一层含义\n\n最后一次季度选择如下；所有 11 个季度均保存在 `parameter_choices.csv`，不是看完整历史后给各币写一组最优参数。\n\n'+table(latest))
    opp=opportunity[(opportunity.phase=='reused_2026')&(opportunity.horizon==24)].groupby('state',as_index=False).agg(平均未来bp=('mean_bp','mean'),合约状态小时数=('n','sum'))
    opp['state']=opp.state.map(REGIME_NAMES)
    add('2026 条件机会（逐币均值再等权，不是独立状态事件）：\n\n'+table(opp.round(2))+'\n\n状态变化可以提供比静态状态中心更有用的交易信息；`transition_future_opportunities.csv` 保存全部 from→to 与下一 24h 分布。当前只是描述，未回灌阈值。状态条件均值若变号，旧原型的未来方向承诺就不应该继续沿用，即使几何路径很相似。')
    add('## 13　费用和组合口径\n\n您给定 maker 单边 2bp 作为主费用、taker 单边 5bp 对照，所有策略均运行两个账本，实际费用改变净值与之后数量。同一模型/动作规则不在 5bp 下另行调参。5m 开盘价格重放不是 maker 成交率模型，这里重点评价策略机制，不推进订单执行工程。\n\n12 币资金费率本地只有 BTC/ETH 覆盖；主比较统一不含 funding，故称价格减手续费。已有两币费率在 `local_funding_sensitivity.csv` 单列，缺失不记为真实零费率。等初始 1/12 子账户，此后权益份额自然变化，组合总净值为子账户权益之和，未免费跨币每小时等权。组合日内方向时间/算术分量为日初资金权重诊断；精确多空利润采用单独资本贡献表。')
    add('## 14　研究取舍与下一步策略实验\n\n继续保留有经济含义的完整家族，而不是用本轮最高收益的单一参数代替主线。优先推进三个问题：\n\n1. **趋势状态的动态持续性**：把方向持续、广度、因子共振与“本币偏离共同市场”放到动作门控；高波且方向一致可以加快参与，高波来回跳则降低对趋势证据的信任。\n2. **空头与反弹的独立价值**：针对 short→exit→rebound 的转换学习动作价值，标签增加 24/72h 继续空与转多差值、逆行路径；避免仅学习平均收益然后共用任意阈值。\n3. **适应速度与持有状态**：专家输出保留“当前仓位/已持有多久/从何状态进入”，不同状态分别学习记忆和退出速度；更长感受野的卷积或小型循环网络与当前直接头比较。\n\n这些是已完成本轮证据后的后续研究方向，尚未冒充本轮已回测模型。新日期按同一冻结算法先生成信号再读取结果；历史仍可研究，但保留其复用性质。短腿质量、跨币覆盖和跨期稳定性共同决定是否升级，不为制造均衡而强迫做空。')
    add('本轮可具体保留的策略原型：慢趋势与通道作为方向机制参照；经济状态/折扣专家作为受限挑战者。区间反转单专家全期为负，不能因“震荡应该均值回复”就配置它；移除反转的门控消融只有小幅差异，不能把它当成全部失败解释。14/42/126 日在线专家结果接近，42 日方案全期几乎没有胜经济门控，当前适应收益不足以宣称已修复漂移。季度择速也未稳定胜固定慢趋势，不能把参数动态化视为天然升级。滚动树和当前直接动作网络保留为失败模型，下一次复杂度增加必须对应上文具体转折问题。')
    add('## 15　可复现入口与边界\n\n```powershell\ncmd /c "call D:\\Total_Tools\\miniforge3\\Scripts\\activate.bat && conda activate universal && python scripts/crypto_beta_strategy_v2.py"\ncmd /c "call D:\\Total_Tools\\miniforge3\\Scripts\\activate.bat && conda activate universal && python scripts/crypto_beta_strategy_v2_diagnostics.py"\ncmd /c "call D:\\Total_Tools\\miniforge3\\Scripts\\activate.bat && conda activate universal && python scripts/crypto_beta_strategy_v2_report.py"\n```\n\n独立配置、20 余维特征登记、11 个季度选择与训练、24 家族全部 12 币的 CSV、当地特征/连续信号/持仓 parquet 已保存。原始数据只读，旧 V1 与形态原型路线不覆盖。基本因果测试与标签成熟检查保留，以避免前向结果失真；本轮没有将执行取数、容量与合约认证作为研究中心。')
    md='\n\n'.join(text)+'\n';(OUT/'MARKET_BETA_STRATEGY_V2.md').write_text(md,encoding='utf-8')
    # HTML concentrates on result discovery and visual comparisons, with full analysis linked separately.
    figures=''.join(f'<figure><img src="{embed(OUT/f"portfolio_{j+1:02}.png")}" alt="{html.escape(t)}"><figcaption>{html.escape(t)} · 最多三条线；灰区为复用 2026</figcaption></figure>' for j,(t,_) in enumerate(GROUPS))
    coinpanes=''.join(f'<div class="coin-pane" data-symbol="{s}" style="display:{"block" if s=="BTCUSDT" else "none"}"><img src="{embed(OUT/f"coin_{s}.png")}" alt="{s} 逐合约曲线"></div>' for s in contracts.symbol.unique())
    payload=scores.replace({np.nan:None}).to_dict('records')
    payload_json=json.dumps(payload,ensure_ascii=False,allow_nan=False).replace('</','<\\/')
    cards=''.join(f'<article class="strategy-card"><span>{i+1:02}</span><h3>{name}</h3><p>全期 {pct(all_returns[p])} · 2026 {pct(reused_returns[p])}</p></article>' for i,(p,name) in enumerate(NAMES.items()))
    metrics=''.join(f'<div class="metric"><small>{NAMES[p]} · 12 币全期</small><strong style="color:{"#b45309" if all_returns[p]<0 else "#0f766e"}">{pct(all_returns[p])}</strong><em>2026：{pct(reused_returns[p])}</em></div>' for p in ['selected_trend','regime_gate','adaptive42','rolling_tree'])
    css='''*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;color:#132c42;background:#f1f5f9;font:16px/1.7 "Microsoft YaHei",sans-serif}.hero{padding:64px max(6vw,24px);color:white;background:linear-gradient(125deg,#081d32,#124c55)}.hero h1{font-size:clamp(30px,4vw,54px);line-height:1.3;max-width:980px;margin:20px 0}.eyebrow{letter-spacing:.12em;color:#67e8cf;font-size:13px}.hero p{max-width:850px;color:#cbdde2}.chips{display:flex;gap:10px;flex-wrap:wrap}.chips span{border:1px solid #41656a;padding:4px 12px;border-radius:30px}.nav{position:sticky;top:0;z-index:3;background:#fff;border-bottom:1px solid #d9e2ea;display:flex;gap:22px;padding:14px 5vw;overflow:auto;font-size:14px}.nav a{color:#155e63;text-decoration:none;white-space:nowrap}main{max-width:1460px;margin:auto;padding:36px 5vw}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}.metric{background:#fff;padding:22px;border-radius:14px;box-shadow:0 5px 22px #132c4208}.metric small,.metric em{display:block;font-style:normal;color:#64748b;font-size:13px}.metric strong{display:block;color:#0f766e;font-size:32px}section{scroll-margin-top:80px;margin-top:42px;background:#fff;padding:32px;border-radius:18px;box-shadow:0 6px 28px #132c4208}h2{font-size:27px;margin:0 0 18px}h3{font-size:18px}.note{background:#eaf5f3;border-left:4px solid #0f766e;padding:20px;margin:20px 0}.formula{font-family:Consolas,monospace;font-size:17px;background:#102b3b;color:#d1fae5;padding:24px;border-radius:12px;overflow:auto}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}figure{margin:0;background:#f8fafc;padding:8px;border-radius:12px}img{width:100%;height:auto;display:block}figcaption{font-size:12px;color:#64748b;padding:10px}.wide{max-width:1160px;margin:25px auto}.controls{display:flex;gap:16px;flex-wrap:wrap;margin:20px 0}label{font-size:13px;color:#64748b}select,button{border:1px solid #cbd5e1;background:#fff;padding:9px 12px;border-radius:8px;color:#153b4a;font:inherit}select{font-size:14px}.table-wrap{overflow:auto;max-height:750px}table{border-collapse:collapse;width:100%;font-size:13px}th{background:#e9f2f3;position:sticky;top:0;text-align:left;white-space:nowrap}td,th{padding:10px;border-bottom:1px solid #e2e8f0}td.num{text-align:right;white-space:nowrap}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:15px}.strategy-card{border:1px solid #e2e8f0;padding:18px;border-radius:12px}.strategy-card span{font:22px Consolas;color:#94a3b8}.strategy-card p{font-size:13px;color:#64748b}.positive{color:#0f766e}.negative{color:#b45309}a{color:#0f766e}.footer{padding:30px;color:#64748b;font-size:13px}.reading{columns:2;column-gap:40px}.reading p{break-inside:avoid}.coin-pane{margin:25px 0}@media(max-width:850px){.metrics,.cards{grid-template-columns:repeat(2,1fr)}.grid{grid-template-columns:1fr}main{padding:25px 15px}section{padding:22px}.reading{columns:1}}@media print{.nav,.controls,button{display:none}section{box-shadow:none;break-inside:avoid}body{background:white}.hero{padding:30px}.grid{grid-template-columns:1fr}.coin-pane{display:block!important}}'''
    css+=' .grid{grid-template-columns:1fr}.grid figure{max-width:1050px;margin:auto;width:100%}'
    body=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>市场 Beta 策略深研 V2</title><style>{css}</style></head><body>
<header class="hero"><div class="eyebrow">MARKET BETA · STRATEGY RESEARCH / V2</div><h1>从市场状态，走到有方向的交易机制</h1><p>12 个 USDT 永续合约 · 24 种策略 · 趋势、空头反弹与非平稳适应。将 HTML 用于比较与可视化，将 Markdown 用于完整分析与策略推导。</p><div class="chips"><span>2026-10-04</span><span>maker 单边 2bp</span><span>taker 单边 5bp</span><span>原始本地行情</span><span>独立方案 A 分支</span></div></header>
<nav class="nav"><a href="#overview">研究判断</a><a href="#portfolio">组合曲线</a><a href="#matrix">12 合约全矩阵</a><a href="#coins">逐合约</a><a href="#legs">多空贡献</a><a href="#beta">择时协方差</a><a href="#models">策略家族</a><a href="#papers">论文与分析</a></nav><main>
<div class="metrics">{metrics}</div><section id="overview"><h2>01｜先经营机会，再定义状态</h2><p>风险水平决定承担多少风险；持续性、下跌速度、反弹恢复与共同市场方向决定承担正 Beta 还是负 Beta。本轮比较状态里该调用哪种交易策略，主动尝试适应模型与直接动作网络。</p><div class="note">全部 24 策略均做全部 12 合约。初始等资金、子账户权益自然变化；主收益口径为价格减手续费，已有 BTC/ETH funding 另列。2026 是复用历史，灰区表示诊断用途，不是新独立 OOS。</div><p><a href="MARKET_BETA_STRATEGY_V2.md">阅读核心分析 Markdown →</a>　<button onclick="window.print()">打印 / 保存 PDF</button></p></section>
<section id="portfolio"><h2>02｜组合表现：按策略问题拆图</h2><p>每图最多三条线，策略家族分开，避免在一张净值图里寻找冠军。财富采用对数轴；maker 2bp，统一不含资金费率。</p><div class="grid">{figures}</div></section>
<section id="matrix"><h2>03｜所有策略 × 所有合约</h2><p>颜色在 −80% 至 +100% 截断以保留辨识度；格内文字保留真实结果。同期币种共振，12 币不是 12 个独立试验。</p><div class="wide"><img src="{embed(OUT/'heatmap_all.png')}" alt="全部策略12合约全期矩阵"></div><div class="wide"><img src="{embed(OUT/'heatmap_reused_2026.png')}" alt="全部策略12合约2026矩阵"></div><div class="controls"><label>策略<select id="policy">{''.join(f'<option value="{p}">{n}</option>' for p,n in NAMES.items())}</select></label><label>阶段<select id="phase"><option value="all">全期 2024–2026</option><option value="development_2024_25">2024–2025</option><option value="reused_2026">2026 复用</option><option value="year2024">2024</option><option value="year2025">2025</option></select></label><label>每边手续费<select id="fee"><option value="2">maker 2bp</option><option value="5">taker 5bp</option></select></label></div><div class="table-wrap"><table><thead><tr><th>合约 / 组合</th><th>累计收益</th><th>日频 Sharpe</th><th>最大回撤</th><th>日频年化波动</th><th>多头时间</th><th>空头时间</th><th>换手加总</th></tr></thead><tbody id="results"></tbody></table></div></section>
<section id="coins"><h2>04｜逐合约：相同规则，独立结果</h2><div class="controls"><label>选择合约<select id="coin">{''.join(f'<option value="{s}" {"selected" if s=="BTCUSDT" else ""}>{s}</option>' for s in contracts.symbol.unique())}</select></label></div><p>每个合约单独呈现趋势/状态与适应/动作。上方矩阵和选择器保留全部策略，不只展示表现好的币种。</p>{coinpanes}</section>
<section id="legs"><h2>05｜空头贡献与持仓期限</h2><p>多空净利润按真实数量组件分开扣费，以该段初始组合资金为分母；两腿加总等于组合该段净利润。更多空头时间并不自动等于更好的负 Beta。</p><div class="wide"><img src="{embed(OUT/'side_contributions.png')}" alt="精确多空净贡献"></div><div class="wide"><img src="{embed(OUT/'holding_duration.png')}" alt="方向持仓期限分布"></div><div class="wide"><img src="{embed(OUT/'cost_comparison.png')}" alt="2bp和5bp费用比较"></div></section>
<section id="beta"><h2>06｜择时协方差：暴露与随后收益的配合</h2><div class="formula">E[B × R_next] = E[B] × E[R_next] + Cov(B, R_next)</div><p>平均暴露项描述漂移；协方差项描述暴露高低或正负与未来市场收益的配合。B 是实际仓位乘事前估计市场 beta，不是当前收益与仓位的事后拟合。剩余项含模型误差，不能自动命名 alpha。</p><div class="wide"><img src="{embed(OUT/'timing_components.png')}" alt="逐币择时协方差分解"></div><div class="note">图中是 5m 算术价格收益加总，和复利收益不同；另存目标暴露的下一小时协方差以区分数量漂移。完整公式、例子与边界在 Markdown §05。</div></section>
<section id="models"><h2>07｜主动尝试复杂模型，保留完整失败结果</h2><div class="cards">{cards}</div><p>季度本币选择只读过去收益；经济状态门控改变策略类型；在线专家遗忘旧关系；隐状态只前向滤波；监督树比较扩展与滚动；小型 TCN 直接学习仓位。模型的具体输入、有效感受野、损失和局限均写入分析文档。</p></section>
<section id="papers"><h2>08｜论文阅读与核心分析</h2><div class="reading"><p><a href="https://www.aqr.com/Insights/Research/Journal-Article/Time-Series-Momentum">Time Series Momentum</a>：本币时序趋势与横截面选币分开。</p><p><a href="https://arxiv.org/html/1607.02410">Trend convexity</a>：趋势保护取决于尺度，波动大小不等于方向机会。</p><p><a href="https://spinup-000d1a-wp-offload-media.s3.amazonaws.com/faculty/wp-content/uploads/sites/3/2019/09/Mom_crashes_JFE_final.pdf">Momentum crashes</a>：跌后反弹改变短腿价值。</p><p><a href="https://business.columbia.edu/sites/default/files-efs/pubfiles/1959/inquire.pdf">Regimes and allocation</a>：前向状态概率服务于动作。</p><p><a href="https://arxiv.org/html/1904.04912">Deep Momentum Networks</a>：方向头与直接动作头分开比较。</p><p><a href="https://arxiv.org/html/1810.10815">Adaptive online learning</a>：学习速度与专家组合应适应变化。</p></div><div class="note"><a href="MARKET_BETA_STRATEGY_V2.md">完整 Markdown 分析</a>：核心认识、24 策略、12 币结果、多空贡献、状态/转移、2026 关系迁移与后续研究设计。详细论文阅读路径保存在仓库 docs。</div></section>
<footer class="footer">本地数据 2023 热身，2024–2025 开发前向，2026 复用诊断，截止 2026-09-25 UTC。无外部 JS/CSS，图表内嵌，可离线查看。策略研究平台，不发送真实订单。</footer></main>
<script>const rows={payload_json};const fmt=x=>(x*100).toFixed(2)+'%';function update(){{const p=document.getElementById('policy').value,phase=document.getElementById('phase').value,fee=Number(document.getElementById('fee').value);document.getElementById('results').innerHTML=rows.filter(x=>x.policy===p&&x.phase===phase&&x.fee_bp===fee).sort((a,b)=>a.symbol.localeCompare(b.symbol)).map(x=>`<tr><td>${{x.symbol}}</td><td class="num ${{x.return_total>=0?'positive':'negative'}}">${{fmt(x.return_total)}}</td><td class="num">${{x.sharpe.toFixed(2)}}</td><td class="num">${{fmt(x.max_dd)}}</td><td class="num">${{fmt(x.vol)}}</td><td class="num">${{fmt(x.long_time)}}</td><td class="num">${{fmt(x.short_time)}}</td><td class="num">${{x.turnover.toFixed(1)}}</td></tr>`).join('');}}['policy','phase','fee'].forEach(id=>document.getElementById(id).addEventListener('change',update));document.getElementById('coin').addEventListener('change',e=>document.querySelectorAll('.coin-pane').forEach(p=>p.style.display=p.dataset.symbol===e.target.value?'block':'none'));update();</script></body></html>'''
    (OUT/'MARKET_BETA_STRATEGY_V2.html').write_text(body,encoding='utf-8')
    (OUT/'report_summary.json').write_text(json.dumps(dict(strategies=len(NAMES),contracts=12,figures=len(list(OUT.glob('*.png'))),html_bytes=len(body.encode()),markdown_bytes=len(md.encode()),all_policy_count=len(manifest['policies']),html_role='offline visual dashboard',markdown_role='core research and mechanisms'),ensure_ascii=False,indent=2),encoding='utf-8')
    print('Reports written',len(body),len(md),flush=True)

if __name__=='__main__':main()
