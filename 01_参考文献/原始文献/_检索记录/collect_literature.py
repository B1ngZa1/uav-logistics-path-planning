"""Reproducible OA literature acquisition. Never modifies existing knowledge cards.
Run with system Python (requests, bs4, pymupdf, Pillow available).
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import csv, hashlib, io, json, re, subprocess, sys, time
import requests
from bs4 import BeautifulSoup
import fitz

sys.stdout.reconfigure(encoding='utf-8')
BASE = Path(__file__).resolve().parent.parent
ROOT = BASE.parent.parent
LOG = BASE / '_检索记录'
SKILL = Path(r'C:\Users\14012\.codex\skills\math-modeling\tools\paper_search\scripts')
CATEGORIES = ['01_RL','02_MARL协同','03_多算法路径规划','04_三维_高度_空中走廊','05_能耗与安全约束','06_综述与基准']
QUERIES = [
 'multi UAV reinforcement learning path planning',
 'multi agent MAPPO cooperative UAV',
 'multi UAV MATD3 trajectory planning',
 'QMIX centralized training decentralized execution',
 'UAV hybrid reinforcement learning path planning',
 'quadrotor three dimensional safe flight corridor',
 'urban air mobility airspace corridor altitude',
 'UAV energy aware trajectory optimization',
 'safe reinforcement learning multi agent barrier',
 'UAV reinforcement learning survey benchmark',
]

# Manually selected from searches; all bibliographic fields are fetched from source pages.
# id, category, arXiv identifier, keywords, project relevance. No scientific results are inferred.
SEEDS = [
 ('RL01',0,'1707.06347','PPO;on-policy;连续动作','单机PPO基线与MAPPO策略更新基础；不是无人机实证。'),
 ('RL02',0,'1801.01290','SAC;最大熵;连续控制','连续速度/航向/爬升控制的off-policy基线；需另加多机约束。'),
 ('RL03',0,'1802.09477','TD3;双Q;延迟更新','理解MATD3的单智能体基础和价值过估计问题。'),
 ('RL04',0,'2311.13081','quadrotor;RL;sim-to-real','低层飞控与轨迹跟踪参考；与高层物流路径规划分层。'),
 ('MA01',1,'2103.01955','MAPPO;CTDE;PPO','多无人机协同训练主要算法基线；原文通用协作游戏不能视为物流实验。'),
 ('MA02',1,'1706.02275','MADDPG;CTDE;集中式critic','建立集中训练、分散执行和局部观测的形式化框架。'),
 ('MA03',1,'1910.01465','MATD3;double centralized critics;连续动作','多智能体双critic方法来源；用于连续三维动作协同基线。'),
 ('MA04',1,'1803.11485','QMIX;值分解;离散动作','航层/路口离散动作候选；连续控制需离散化或分层，注意单调性限制。'),
 ('MA05',1,'2401.01481','MAPPO;MADDPG;UAV-UGV;聚类','任务聚类与MARL联合规划的异构协同参考；核对地空协同与纯空中场景差异。'),
 ('PA01',2,'1809.03531','PRIMAL;模仿学习;强化学习;MAPF','比较规划专家示范与RL融合；离散MAPF向三维动力学迁移需另验证。'),
 ('PA02',2,'2008.08835','EGO-Planner;梯度优化;B-spline','为高层RL输出航点配套局部平滑规划与动力学检查。'),
 ('PA03',2,'2404.01752','SI-RRT*;CBS;优先级规划','采样规划和多机器人冲突消解组合，用作时间窗航道基线。'),
 ('PA04',2,'2007.13065','覆盖路径;VRP;BRKGA;3D mesh','多算法融合与三维环境建模参考；巡检覆盖问题不同于配送。'),
 ('CO01',3,'1909.02896','RSFC;SFC;Bernstein;QP','构造机间相对安全走廊与三维碰撞约束；区分局部安全管道与城市航路。'),
 ('CO02',3,'2208.06950','temporal safe corridor;动态障碍;时空占用','把航道占用和障碍运动纳入时间维；支持路口预约/换层冲突建模。'),
 ('CO03',3,'2301.09758','UAM;DDPG;airspace capacity','城市低空容量和禁入区域约束参考；不能直接提供中国法定高度。'),
 ('CO04',3,'2209.09447','quadrotor swarm;deadlock-free;MAPP','多机狭窄通道死锁与子目标规划；用于层间连接通道调度。'),
 ('SA01',4,'1705.10528','CPO;CMDP;安全强化学习','建立约束MDP与期望代价约束；期望约束不能替代逐时刻无碰撞保证。'),
 ('SA02',4,'2101.05436','CBF;decentralized;安全证书','分布式安全过滤器与多机避碰约束；保证依赖模型和证书条件。'),
 ('SA03',4,'1804.02238','rotary-wing;推进能耗;SCA','旋翼机推进功率建模基础；三维爬升、载荷和风场需校准扩展。'),
 ('RE01',5,'2108.06266','safe RL;MPC;CBF;综述','厘清安全学习、鲁棒控制和可验证安全的区别，支持综述结构。'),
 ('RE02',5,'2109.06325','safe-control-gym;benchmark;constraints','比较学习控制与传统控制的基准；原环境含1D/2D四旋翼而非完整三维物流。'),
 ('RE03',5,'2103.02142','gym-pybullet-drones;multi-agent;simulation','多无人机动力学仿真基准参考，关注控制频率与训练配置可复现。'),
 ('RE04',5,'1911.10635','MARL;理论;CTDE;综述','多智能体强化学习理论和分类基础；为算法选择提供术语对照。'),
 ('RE05',5,'2502.12875','DRL;UAV;communications;survey','通信和轨迹联合优化综述；通信指标不能等同于配送效率。'),
 ('RE06',5,'2301.00912','UAV swarm;distributed learning;MARL','无人机集群分布式学习、通信与感知联合设计背景。'),
]

def save_json(path, obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def normalize(s):
    return ''.join(c.lower() for c in s if c.isalnum())

def snapshot():
    p=LOG/'existing_files_before.json'
    if p.exists(): return
    result={}
    for f in BASE.parent.rglob('*'):
        if f.is_file() and BASE not in f.parents:
            if f.suffix.lower() in ('.md','.csv','.pdf'):
                result[str(f.relative_to(ROOT))]={'size':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
    save_json(p,result)

def run_searches():
    for i,q in enumerate(QUERIES,1):
        target=LOG/f'search_{i:02d}.json'
        if target.exists(): continue
        p=subprocess.run([sys.executable,str(SKILL/'hybrid_scholar.py'),'--query',q,'--limit','12','--json'],capture_output=True,text=True,encoding='utf-8',timeout=100)
        raw=p.stdout
        at=raw.find('{\n')
        try: data=json.loads(raw[at:]) if at>=0 else {}
        except Exception: data={}
        save_json(target,{'query':q,'timestamp':datetime.now().isoformat(),'exit_code':p.returncode,'diagnostics':raw[:at] if at>=0 else raw,'stderr':p.stderr,'result':data})
        print('SEARCH',i,data.get('stats',{}),flush=True)

def request(url):
    return requests.get(url,timeout=(15,45),headers={'User-Agent':'UAV-Literature-Research/1.0 (academic metadata and open-access retrieval)'})

def fetch_arxiv(seed):
    key,cat,aid,kw,rel=seed
    cached=LOG/f'{key}_record.json'
    if cached.exists(): return json.loads(cached.read_text(encoding='utf-8'))
    url=f'https://arxiv.org/abs/{aid}'
    row={'id':key,'category':CATEGORIES[cat],'arxiv':aid,'keywords':kw,'relevance':rel,'source_page':url,'sources':['arxiv'],'cross_validated':False,'doi':'','authors':[],'year':'','title':'','venue':'arXiv','volume':'','issue':'','pages_biblio':'','pdf_path':'','pdf_pages':0,'pdf_status':'未下载','verification':'待核验','download_attempts':[],'timestamp':datetime.now().isoformat()}
    try:
        r=request(url); r.raise_for_status()
        soup=BeautifulSoup(r.text,'html.parser')
        def meta(name):
            return [x.get('content','') for x in soup.find_all('meta',attrs={'name':name})]
        row['title']=(meta('citation_title') or [''])[0]
        row['authors']=meta('citation_author')
        row['year']=(meta('citation_date') or [''])[0][:4]
        row['doi']=(meta('citation_doi') or [''])[0]
        row['arxiv_version']=(meta('citation_arxiv_id') or [aid])[0]
        journal=soup.select_one('.jref')
        row['journal_reference']=journal.get_text(' ',strip=True) if journal else ''
        abstract=soup.select_one('.abstract')
        row['abstract']=abstract.get_text(' ',strip=True) if abstract else ''
        row['verification']='arXiv官方页作者/题名/首次提交年份已核验；预印本，不推定正式出版信息'
        row['official_http']=r.status_code
        row['page_evidence']=str((LOG/f'{key}_source.html').relative_to(BASE))
        (LOG/f'{key}_source.html').write_text(r.text,encoding='utf-8')
        if not row['title'] or not row['authors']: raise ValueError('官方页面缺少题名或作者')
        pdfurl=(meta('citation_pdf_url') or [f'https://arxiv.org/pdf/{aid}'])[0]
        row['pdf_url']=pdfurl
        filename=f'{key}_{row["year"]}_{aid}.pdf'
        dest=BASE/CATEGORIES[cat]/filename
        if not dest.exists():
            pr=request(pdfurl)
            row['download_attempts'].append({'url':pdfurl,'http':pr.status_code,'bytes':len(pr.content)})
            pr.raise_for_status()
            if not pr.content.startswith(b'%PDF'): raise ValueError('响应不是PDF')
            # Validate bytes before saving, never save a failed web page as PDF.
            doc=fitz.open(stream=pr.content,filetype='pdf')
            if doc.page_count<1: raise ValueError('PDF无页面')
            doc.close()
            dest.write_bytes(pr.content)
        row['pdf_path']=str(dest.relative_to(ROOT))
        row['pdf_status']='已下载arXiv公开版本'
    except Exception as e:
        row['error']=str(e)
        if not row['pdf_path']: row['pdf_status']='下载或元数据失败；保留来源链接'
    save_json(cached,row)
    print('ARXIV',key,row['title'][:75],row['pdf_status'],flush=True)
    return row

def validate(row):
    if not row.get('pdf_path'): return row
    path=ROOT/row['pdf_path']
    try:
        with fitz.open(path) as doc:
            chars=[len(p.get_text().strip()) for p in doc]
            row['pdf_pages']=len(doc)
            row['text_chars_total']=sum(chars)
            row['text_pages']=sum(n>=40 for n in chars)
            first=doc[0].get_text()
            row['title_pdf_match']=normalize(row['title'])[:70] in normalize(first)
            row['pdf_readability']='逐页文本提取成功' if sum(chars)>500 and row['text_pages']>=max(1,len(doc)//2) else '需人工核验：文本稀少'
            preview=LOG/'预览'/f'{row["id"]}_first.png'
            preview.parent.mkdir(exist_ok=True)
            doc[0].get_pixmap(matrix=fitz.Matrix(0.9,0.9)).save(preview)
            row['preview']=str(preview.relative_to(BASE))
            # Render every page at low resolution to ensure pages are parseable.
            for pg in doc: pg.get_pixmap(matrix=fitz.Matrix(.15,.15),alpha=False)
            row['render_check']='全部页面低分辨率渲染成功；首页另存预览'
        row['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        row['bytes']=path.stat().st_size
    except Exception as e:
        row['pdf_readability']='核验失败: '+str(e)
    save_json(LOG/f'{row["id"]}_record.json',row)
    return row

def main():
    for c in CATEGORIES: (BASE/c).mkdir(parents=True,exist_ok=True)
    snapshot()
    if '--search' in sys.argv: run_searches(); return
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows=list(pool.map(fetch_arxiv,SEEDS))
    for row in rows: validate(row)
    save_json(LOG/'arxiv_records.json',rows)
    print('TOTAL',len(rows),'DOWNLOADED',sum(bool(r['pdf_path']) for r in rows),flush=True)

if __name__=='__main__': main()
