生醫文獻全文檢索－完整可執行版

2026-10-04 更新：PubMed / arXiv 雙來源匯入

更新既有專案：關閉網站，覆蓋 app.py、engine.py、collection_store.py，
再加入新的 arxiv_bulk.py。README_請先看.txt 可一起更新。
請勿覆蓋原本的 data 資料夾。舊 PubMed 主題、文章和刪除紀錄會保留。

大量匯入：到 Manage Articles，先選 Source（PubMed 或 arXiv），
再填主題名稱、搜尋條件與篇數（10–1,000，預設 500），不需自行尋找 ID。
主題名稱會產生對應來源的 title/abstract 查詢；手動改過後不會被自動覆蓋。
arXiv 可使用分類查詢，例如 cat:cs.AI。NCBI 聯絡信箱僅在 PubMed 顯示。
切換來源時各自保留搜尋條件；追加批次依來源和查詢分別記錄頁碼。

指定文章：按 Add Articles，先選 Source，再貼上多筆 PMID 或 arXiv ID。
一次最多 200 個 ID，支援空白、逗號、分號或換行。
arXiv 支援 2010.11929、1706.03762、hep-th/9901001，以及 arxiv.org 的
/abs/ 和 /pdf/ 連結。版本號會移除，同一篇的不同版本不會重複加入。
已有或先前刪除的文章不會重新加入。

arXiv 僅匯入 title、abstract 與識別資料，不下載 PDF 或全文。
內部 ID 使用 ARXIV:2010.11929；PubMed 仍使用 PMID 前綴，避免兩者衝突。
搜尋、Zipf A–D、Compare Domains、Word2Vec、PCA 與 CSV 匯出沿用原流程。
匯入需要可連線至 export.arxiv.org 或 eutils.ncbi.nlm.nih.gov。
arXiv API 呼叫有間隔與有限次重試；若服務暫時無法使用，會顯示失敗原因，
不會為此清除既有文章。跨領域比較請使用相近篇數，並於報告記錄兩個來源。

這一包已經包含所有必要程式，不需要和任何舊版合併：

  app.py       網站與介面
  engine.py    XML 解析、統計、搜尋與 BM25
  zipf_analysis.py  Project 2 的 CF、DF、IDF、Zipf 與迴歸分析
  word2vec_model.py  Project 2 的 CBOW／Skip-gram Word2Vec
  spelling.py  Dynamic Programming Edit Distance 拼字修正
  collection_store.py  多主題資料集管理
  porter.py    Porter stemming
  pmc.py       PMID 轉 PMCID，以及下載 PMC XML
  pubmed_bulk.py  依主題批次下載 PubMed 摘要
  arxiv_bulk.py  依主題或 arXiv ID 批次下載 arXiv 摘要
  data/        網站匯入資料的保存位置（初始為空）
  啟動網站.bat  Windows 快速啟動

執行方式：

1. 對 ZIP 按右鍵，選「解壓縮全部」。
2. 進入解壓縮後的 iir_project1 資料夾。
3. 在資料夾空白處按 Shift＋滑鼠右鍵，選「在終端機中開啟」。
4. 輸入：

       python app.py

   如果電腦無法使用 python 指令，改輸入：

       py app.py

   也可以直接雙擊「啟動網站.bat」。

5. 瀏覽器開啟：

       http://127.0.0.1:8765

6. 關閉網站時，回到終端機按 Ctrl＋C。

如果你的舊資料夾已經匯入文章，更新時只需覆蓋程式檔，並保留原本的 data 資料夾；
不要用新壓縮檔內的空 data 資料夾覆蓋已匯入的文章。

目前統計規則：

- Words、Sentences、Characters 只計算摘要。
- 搜尋詞顯示摘要、正文與全文三種次數；單一詞另外顯示摘要 collection 的 CF、DF、IDF。
- 搜尋範圍包含摘要與正文，不包含標題。
- 標題只顯示，不參與搜尋，也不會因查詢而反白。
- 小數視為一個 word；例如 0.65 是一個，0.44–0.97 是兩個。
- 左側可用標題、PMCID 或固定編號即時搜尋已載入的文章。
- 點文章標題會進入該篇文章的單篇搜尋模式。
- 文章依自訂主題分組，例如 GLP-1、Brain cancer；搜尋與分析只使用目前選定的主題。
- 可在既有主題追加下一批 PubMed 結果，系統會記錄查詢位移並略過重複 PMID。
- 可刪除整個主題，也可從文章清單逐篇刪除；刪除後會立即重建分析結果。
- 文章詳細頁顯示摘要，下面提供 PMC 原文連結。
- testing、tested、tests 會經 Porter stemming 歸為 test。
- xref 引用標籤會補上文字邊界，避免 testing10 黏成一個詞。

注意：

- 執行時不要刪除 pmc.py。
- data 資料夾會保存網站匯入的 XML／JSON 與文章編號。
- PMID 查詢與 PMC 原文連結需要網路。
- 程式只使用 Python 內建套件，不需要 pip install。

Project #2 新增功能：

1. 網站上方新增「Zipf／詞彙分布」頁面。
2. Zipf 頁只使用 abstract，標題與正文不計入 Project #2 分析。
3. 比較四種累積 preprocessing：
   A：空白斷詞＋小寫
   B：移除標點並使用專案 tokenizer
   C：移除 stopwords
   D：Porter stemming
4. 顯示 documents、total tokens、unique terms、average tokens/document。
5. 顯示 Rank vs Frequency 與 Log-Log Plot。
6. 線性迴歸會報告 slope、intercept、Zipf exponent、R²、RMSE。
7. 顯示 Top 50 terms 的 CF、DF、IDF，並可下載完整 CSV。
8. 「單一詞分布搜尋」可查一個詞在每篇摘要中的 TF，並顯示整體 CF、DF、IDF。結果會在獨立頁面顯示，可在頁面頂端直接改查其他單詞。
9. 一般文獻搜尋若只輸入一個詞，結果頁也會直接顯示摘要 collection 的 CF、DF、IDF，並連到完整分布。
10. 除 PMC JATS 全文 XML 外，現在也可上傳 PubMed EFetch XML；一個 XML 可以包含多篇 PubmedArticle records。
11. 也可直接上傳先前 Project #2 下載程式產生的 data/pubmed.json，不必重新下載或轉檔。
12. 「PMID查詢/上傳」可輸入 PubMed 主題並批次加入 10～1,000 篇摘要；欄位只顯示
    GLP-1[Title/Abstract] 作為提示範例，不會自動填入，結果依 PubMed relevance 排序，重複 PMID 會自動略過。
13. Zipf 頁的條件按鈕、統計表與 CSV 下載已合併在同一區塊，圖表圖例顯示於右上角。
14. 支援多個命名主題；每個主題的 Zipf、CF、DF、IDF 與搜尋索引完全分開。
15. 上方第三個頁面為「Word2Vec模型」，第四個頁面為「Zipf分布圖」。
16. Word2Vec 可選 CBOW 或 Skip-gram，兩者都使用 Negative Sampling，並可查相似詞。
17. 可調整 window size（1～5）、vector dimension（8～64）與 epochs（1～5）；預設為 2、24、2。
18. 搜尋加入 Dynamic Programming Levenshtein Edit Distance 拼字修正，會顯示「你是不是要找……」。
19. 管理頁可切換主題、在既有主題追加文章，或一次刪除整個主題。
20. 建立新主題時，主題名稱與 PubMed 搜尋條件預設為空白，仅顯示範例提示詞，不再預先填入 GLP-1。
21. 頁面上方的「加入文章」會開啟小視窗，可一次貼上最多 200 個 PMID 批次加入摘要，或上傳一篇 PMC JATS XML／NXML。
22. 已有文章清單可以逐篇刪除；刪除後會立即重建編號、搜尋索引、Zipf 與 Word2Vec 分析。
23. 舊的單一 PMID 查詢區已移除；PMID 匯入統一放在「加入文章」視窗。Zipf 頁首次開啟時預設顯示 A（空白斷詞＋小寫）。
24. 建立新主題時，輸入主題名稱會自動產生「主題[Title/Abstract]」PubMed 搜尋條件；若手動修改搜尋條件，系統會保留自訂內容。
25. Zipf 頁的 Collection Vocabulary 旁新增醒目的 Compare Domains，可任選兩個既有主題，比較 Documents、Tokens、Vocabulary、Zipf exponent、R²、RMSE、Top 10 terms 與重疊 Log-Log 曲線。
26. PubMed 匯入篇數、批次 PMID 與 Word2Vec 參數欄位下方會以英文標示最大值。
27. 全站主要操作按鈕與導覽改為英文，統一使用低彩度深青綠、細框與輕量滑鼠回饋；刪除操作保留紅色警示樣式。
28. 主題管理卡片將文章篇數放在主題名稱右側，使用較大的醒目標籤呈現。
29. Word2Vec 訓練結果新增互動式 PCA 詞向量分布圖；滑鼠移到資料點可查看詞、PC1、PC2、corpus frequency 與 cosine similarity。PCA 使用純 Python 計算，不需額外安裝套件。
30. Zipf 的線性與 log-log 圖新增滑鼠資料點提示，可查看 word、rank、frequency 與 log 座標；頁面移除重複的 collection 統計卡及非必要的高／中／低頻區段表格。
31. 主題管理頁的每個主題卡片皆為獨立區塊，不會再出現下一個主題被包在上一個主題內的巢狀排版。

新版作業要求約 1,000 篇英文 scientific abstracts。初始資料集為空；
請在「PMID查詢/上傳」直接依主題下載摘要，或上傳自己的 PubMed XML／pubmed.json collection。系統會略過沒有 abstract 的 PubMed record，
並以 PMID 作為唯一 Document ID。Zipf 頁若文件不足 900 篇會顯示提醒。

老師要求的報告內容仍需另外完成：

- 回答 RQ1～RQ5 與各 Part 的分析問題。
- 撰寫 300～500 words：Why does Zipf’s Law matter to an Information Retrieval system?
- 另外繳交一頁 Executive Summary。
