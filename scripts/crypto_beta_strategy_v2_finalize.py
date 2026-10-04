"""Record the completed research, without changing strategy decisions."""
import hashlib,json,re,shutil,subprocess
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/crypto/market_beta_strategy_v2'

def main():
    scores=pd.read_csv(OUT/'strategy_scores.csv');diag=json.loads((OUT/'diagnostic_summary.json').read_text())
    html=(OUT/'MARKET_BETA_STRATEGY_V2.html').read_text(encoding='utf-8')
    md=(OUT/'MARKET_BETA_STRATEGY_V2.md').read_text(encoding='utf-8')
    assert len(scores)==3120
    assert len(scores.policy.unique())==24 and len(diag['all_12_contracts'])==12
    assert scores.groupby(['policy','phase','fee_bp']).symbol.nunique().eq(13).all()
    assert html.count('data:image/png;base64,')==27
    assert 'src="http' not in html and 'stylesheet" href="http' not in html
    assert diag['forward_maturity'] and diag['max_leg_accounting_error']<1e-10
    assert len(re.findall(r'^## ',md,re.M))==15
    assert '86 passed' in (OUT/'tests.log').read_text(encoding='utf-8')
    node=shutil.which('node')
    if node:
        script=re.search(r'<script>(.*?)</script>',html,re.S).group(1)
        result=subprocess.run([node,'--check','-'],input=script,text=True,capture_output=True)
        if result.returncode:raise RuntimeError(result.stderr)
    roots=[ROOT/'configs/crypto_beta_strategy_v2.json',ROOT/'src/crypto/beta_strategy.py',ROOT/'tests/test_crypto_beta_strategy.py',ROOT/'docs/MARKET_BETA_STRATEGY_V2_PLAN.md',ROOT/'docs/MARKET_BETA_STRATEGY_V2_LITERATURE.md']
    roots+=list((ROOT/'scripts').glob('crypto_beta_strategy_v2*.py'))
    roots+=list(OUT.glob('*'))
    hashes={}
    for p in roots:
        if p.is_file() and p.name!='research_manifest.json':
            b=p.read_bytes()
            if p.suffix in ['.py','.json','.md','.csv','.html','.log']:b=b.replace(b'\r\n',b'\n')
            hashes[str(p.relative_to(ROOT)).replace('\\','/')]=hashlib.sha256(b).hexdigest()
    (OUT/'research_manifest.json').write_text(json.dumps(dict(recorded_on='2026-10-04',branch='codex/market-beta-a',strategy_count=24,contract_count=12,fee_scenarios=[2,5],figure_count=27,source_hashes=hashes,historical_status='development_forward_and_reused_2026',label_maturity=True,leg_identity_max_error=diag['max_leg_accounting_error'],text_hash_rule='canonical LF; binary exact',tests='86 passed',strategy_selection_status='no live certification; all outcomes retained'),ensure_ascii=False,indent=2),encoding='utf-8')
    state='''# 项目状态

更新：2026-10-04 · `codex/market-beta-a` 独立分支 · v0.4.0 策略深研 V2。

## 当前用户偏好与入口

当前优先研究策略本质、方向 Beta、趋势持续性、下跌/反弹转换与非平稳适应；执行取数与繁重检验不作为主工作。全部 12 USDT 永续合约都做，maker 单边 2bp 为主、taker 5bp 对照；持仓接受十几小时到十多天。名义多空目标上限对称，利润和时间不强凑 50/50。原形态原型和未来其他方法继续保留，旧产物不覆盖。

- [HTML 可视化仪表板](reports/crypto/market_beta_strategy_v2/MARKET_BETA_STRATEGY_V2.html)：27 张拆分图、逐合约选择与完整结果表。
- [Markdown 核心分析](reports/crypto/market_beta_strategy_v2/MARKET_BETA_STRATEGY_V2.md)：论文、公式、机制、结果与失败解释。
- [研究计划](docs/MARKET_BETA_STRATEGY_V2_PLAN.md) · [原始论文阅读](docs/MARKET_BETA_STRATEGY_V2_LITERATURE.md)。
- [前轮 V1 报告](reports/crypto/market_beta_a_v1/MARKET_BETA_A_REPORT.md)及完整原型历史均保留。

## 本轮已完成

1. 读取趋势凸性、动量崩盘、状态配置、直接仓位网络、适应在线学习原文相关章节；摘要与全文访问边界单列。
2. 24 策略全部运行 12 合约：四速度趋势、通道、季度本币选择、Kalman、区间/跌势反弹专家、经济门控、14/42/126 日条件在线专家、四隐区滤波专家、扩展/滚动多期限树及直接动作 TCN；多头对照齐全。
3. 季度前向 11 折。完整小时特征 t 可用、t+5m 执行，监督最长 240h 成熟后训练；在线 1h 成交收益至 t+2h 才可更新。基本因果测试保留，全项目 86 项通过。
4. 初始等资金的独立子账户，权益份额自然变化，组合不免费再平衡。全部策略各运行 maker 2bp / taker 5bp。主口径统一价格减手续费；BTC/ETH 已有本地 funding 另列。
5. 精确多空净资本贡献、逐币实际/目标择时协方差、状态/转移机会、方向持仓期、参数选择与信号迁移全部保存；资本贡献恒等式最大误差约 2.4e-13。

## 核心策略结论

2bp 组合全期 / 2026 复用：168h 趋势 +38.39/+22.30%，通道 +64.04/+13.08%，经济门控 +27.12/+10.12%，42 日专家 +27.05/+11.22%，季度本币择速 +21.18/+14.51%。新口径与 V1 不直接相减。

168h 趋势 2026 空头净贡献 +13.02%，多头 +9.28%；全期空头 −4.97%。多空时间约各 47–49%，阶段利润仍不均衡，这是机会与关系问题，不能只加短仓。

扩展树 / 滚动树 / TCN 的 2026 为 −12.45/−31.22/−27.52%；滚动与 TCN 的信号对随后24h关联在11/12、10/12币为负。简单缩窗未修复迁移；区间反转也失败。14/42/126日遗忘差异小，适应增量未证明。隐状态编号可重学，不宣称固定交易原型。

## 下一步：继续策略深度研究

- 慢趋势与通道作为方向机制参照，经济状态/适应专家作为受限挑战者；全部失败模型继续保留，不按最高净值挑认证赢家。
- 研究趋势持续性、共同因子一致性与本币偏离；在高波持续趋势和高波来回跳之间改变策略类型与速度。
- 针对 short→exit→rebound 学习继续持空与转多的动作价值，补条件逆行路径；不只预测均值后套固定阈值。
- 将当前持仓、持有时长、进入状态放入动作模型；比较更完整时间感受野与多记忆模型。当前 TCN 最后读出仅有效12h时序感受野，已明确记录。
- 2024–25为开发前向、2026为复用诊断；新日期预测先存再读标签，才形成新的顺序证据。当前没有新独立 OOS，不能凭历史修复重新认证。

## 交付状态

源码、配置、CSV（逐币日表 gzip）、模型权重、27张图、分工不同的两个报告与日志在本分支。当地特征/信号/持仓 parquet 与原始行情不上传 Git。GitHub SSH 已可推送；已有集成的 PR 创建权限为403，独立分支交付不合并主线。
'''
    (ROOT/'STATE.md').write_text(state,encoding='utf-8')
    log='''

## 2026-10-04 · C-E015 方案 A 策略机制深研 V2（实际运行）

- 问题：12 币是否存在可持续正/负 Beta 方向机会？短腿贡献不足是否来自上限、速度或跌后反弹错配？滚动/条件专家能否改善旧关系迁移？
- 方法：原始论文机制阅读；24 家族×12 合约×2费用，11季度前向时间规则；四速度趋势/通道、本币过去选择、Kalman、经济门控、14/42/126日在线专家、滤波隐区专家、扩展/滚动24/72/240h树与4h直接TCN。精确方向资本贡献与平均因子暴露/择时协方差分别解释。
- 数据：当地2023-01—2026-09-24，完整1h可知，下一5m执行；2023热身，2024–25开发前向，2026复用。用户maker单边2bp主情景、taker5bp对照；主表不含缺失funding，BTC/ETH本地费率单列。
- 结果：全期/2026慢趋势+38.39/+22.30%，通道+64.04/+13.08%，经济门控+27.12/+10.12%，42日专家+27.05/+11.22%。慢趋势2026空头+13.02%、多头+9.28%，全期空头−4.97%。滚动树/TCN2026−31.22/−27.52%，信号未来关联在11/12、10/12币为负，短窗与复杂化未自动修复。状态门控和季度选择未稳定胜朴素机制。
- 限制：全部历史已被探索，不改称未触碰OOS；当前12币面板有选择偏差；监督目标/固定映射及小TCN感受野/持仓状态有错配。maker费率并非被动成交模型；10币funding缺失不当真实零。14/42/126遗忘结果接近，不宣称适应增量已成立。
- 必要核查：11折标签完全成熟，4新增因果测试，全项目86通过；多空净资本贡献恒等式最大误差2.4e-13。未展开新执行取数、容量或合约认证工程。
- 产物：独立源码/配置、全量正反CSV与当地parquet、27张最多3曲线子图、HTML结果仪表板、Markdown核心分析、论文阅读与策略计划。
- 下一步：以方向持续性/共同因子一致性和短空→退出→反弹动作价值为核心，加入持仓时长/进入状态与更完整时序表示；少数完整家族新日期先留档再评价，原型其他路线继续保留。
'''
    if 'C-E015 方案 A 策略机制深研 V2（实际运行）' not in (ROOT/'RESEARCH_LOG.md').read_text(encoding='utf-8'):
        with (ROOT/'RESEARCH_LOG.md').open('a',encoding='utf-8') as f:f.write(log)
    print('Recorded',len(hashes),'research artifacts; strategies=24 contracts=12',flush=True)

if __name__=='__main__':main()
