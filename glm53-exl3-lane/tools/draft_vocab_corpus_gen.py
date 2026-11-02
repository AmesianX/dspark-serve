"""Generate domain answers (gukhanmun / English / coding / IT) from the running server for the
draft-vocab corpus, train and held-out eval prompts kept separate. Reasoning text is included
because the drafter predicts it too. Usage: python3 tools/draft_vocab_corpus_gen.py <workdir>/gen.json
"""
import json, urllib.request, concurrent.futures as cf, sys
URL="http://127.0.0.1:8000/v1/chat/completions"; import _lane; KEY=_lane.api_key()
NEWS="政府는 今日 國務會議에서 第三次 經濟開發五個年計劃의 年度別 投資計劃을 議決하고 輸出目標額을 十億弗로 確定하였다."
P={
"gukhanmun":{"train":[
 "1970년대 신문 사설처럼 國漢文混用體로 農村 近代化에 대한 論說을 써줘.",
 f"다음 國漢文 記事를 現代 한국어로 풀어 쓰고 漢字語를 풀이해줘: {NEWS}",
 "國漢文混用으로 1960년대 서울의 都市化 問題를 다룬 新聞 解說 記事를 작성해줘.",
 "漢字成語 20개를 漢字 表記와 함께 뜻과 由來를 설명해줘.",
 "朝鮮時代 科擧制度를 國漢文混用體로 설명하는 글을 써줘."],
 "eval":[
 "國漢文混用體로 敎育熱과 大學入試制度에 관한 1970년대 新聞 論說을 써줘.",
 "다음 문장을 國漢文混用體로 바꿔줘: 정부는 물가 안정을 위해 공공요금 인상을 억제하기로 결정하였다.",
 "四字成語 15개를 漢字와 함께 例文을 들어 설명해줘."]},
"english":{"train":[
 "Write an essay on the causes of the 2008 financial crisis.",
 "Explain how the immune system distinguishes self from non-self.",
 "Summarize the history of the Cold War in Europe from 1947 to 1991.",
 "Explain the central limit theorem with an intuitive example.",
 "Describe how CRISPR-Cas9 gene editing works."],
 "eval":[
 "Explain the geopolitical significance of the Strait of Malacca.",
 "Write an explanation of entropy in thermodynamics and information theory.",
 "Describe the causes and consequences of the fall of the Western Roman Empire."]},
"coding":{"train":[
 "Python으로 LRU 캐시를 구현하고 테스트 코드도 작성해줘.",
 "Write a Rust function that parses a CSV file into structs with error handling.",
 "C로 연결 리스트 기반 해시맵을 구현해줘. 충돌 처리 포함.",
 "Write a TypeScript React component for a paginated table with sorting.",
 "Bash 스크립트로 디렉터리의 로그 파일을 압축하고 30일 지난 것은 삭제하는 스크립트를 짜줘."],
 "eval":[
 "Go로 동시성 있는 웹 크롤러를 구현해줘. 최대 동시 요청 수 제한 포함.",
 "Write a SQL schema and queries for a library management system.",
 "Python asyncio로 TCP 에코 서버와 클라이언트를 작성하고 설명해줘."]},
"it_terms":{"train":[
 "쿠버네티스의 Pod, Deployment, Service, Ingress의 차이를 설명해줘.",
 "Explain TCP congestion control algorithms: Reno, CUBIC, and BBR.",
 "RDMA와 RoCE v2의 동작 원리를 설명해줘.",
 "Explain how a CPU cache hierarchy and cache coherence protocols work.",
 "CUDA의 워프, 블록, 그리드, 공유 메모리 개념을 설명해줘."],
 "eval":[
 "Explain how Linux virtual memory, page tables and TLB work.",
 "데이터베이스 트랜잭션 격리 수준과 MVCC를 설명해줘.",
 "Explain how DNS resolution works end to end."]}}
def ask(q):
    b=json.dumps({"model":"glm-5.3-flash","messages":[{"role":"user","content":q}],"max_tokens":1400,"temperature":1.0}).encode()
    r=urllib.request.Request(URL,b,{"Content-Type":"application/json","Authorization":"Bearer "+KEY})
    d=json.loads(urllib.request.urlopen(r,timeout=900).read())
    m=d["choices"][0]["message"]
    return (m.get("reasoning_content") or m.get("reasoning") or "")+"\n"+(m.get("content") or "")
jobs=[(dom,split,q) for dom,v in P.items() for split in ("train","eval") for q in v[split]]
out=[]
with cf.ThreadPoolExecutor(4) as ex:
    futs={ex.submit(ask,q):(dom,split,q) for dom,split,q in jobs}
    for f in cf.as_completed(futs):
        dom,split,q=futs[f]
        try: t=f.result()
        except Exception as e: print("ERR",dom,e,flush=True); continue
        out.append({"domain":dom,"split":split,"prompt":q,"text":t}); print(f"done {len(out)}/{len(jobs)} {dom}/{split} {len(t)} chars",flush=True)
json.dump(out,open(sys.argv[1],"w"),ensure_ascii=False)
