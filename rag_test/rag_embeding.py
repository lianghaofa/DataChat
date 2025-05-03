import re
import numpy as np
import trio
import tiktoken
from milu_cnn import milvus_con

encoder = tiktoken.get_encoding("cl100k_base")
def truncate(string: str, max_len: int) -> str:
    """Returns truncated text if the length of text exceed max_len."""
    return encoder.decode(encoder.encode(string)[:max_len])

async def embedding(docs, mdl, parser_config=None, callback=None):
    if parser_config is None:
        parser_config = {}
    batch_size = 16
    tts, cnts = [], []
    for d in docs:
        tts.append(d.get("docnm_kwd", "Title"))
        c = "\n".join(d.get("question_kwd", []))
        if not c:
            c = d["content_with_weight"]
        c = re.sub(r"</?(table|td|caption|tr|th)( [^<>]{0,12})?>", " ", c)
        if not c:
            c = "None"
        cnts.append(c)

    tk_count = 0
    if len(tts) == len(cnts):
        vts, c = await trio.to_thread.run_sync(lambda: mdl.encode(tts[0: 1]))
        tts = np.concatenate([vts for _ in range(len(tts))], axis=0)
        tk_count += c

    cnts_ = np.array([])
    for i in range(0, len(cnts), batch_size):
        vts, c = await trio.to_thread.run_sync(lambda: mdl.encode([truncate(c, mdl.max_length-10) for c in cnts[i: i + batch_size]]))
        if len(cnts_) == 0:
            cnts_ = vts
        else:
            cnts_ = np.concatenate((cnts_, vts), axis=0)
        tk_count += c
        # callback(prog=0.7 + 0.2 * (i + 1) / len(cnts), msg="")
    cnts = cnts_

    title_w = float(parser_config.get("filename_embd_weight", 0.1))
    vects = (title_w * tts + (1 - title_w) *
             cnts) if len(tts) == len(cnts) else cnts

    assert len(vects) == len(docs)
    vector_size = 0
    for i, d in enumerate(docs):
        v = vects[i].tolist()
        vector_size = len(v)
        d["q_%d_vec" % len(v)] = v
    return tk_count, vector_size

async def main():
    from embeding_model import SILICONFLOWEmbed
    model=SILICONFLOWEmbed(key="sk-ymdafgvllilgrwobjenyzeersaefeaoyrxyeptpbyfnuddeh",model_name="BAAI/bge-m3")
    from table import chunk
    file_path="./data.xlsx"
    chunks=chunk(filename=file_path,from_page=0,to_page=100)
    token_count, vector_size= await embedding(docs=chunks,mdl=model)
    doc_store_result = ""
    es_bulk_size = 4
    milvus_con.createIdx("test", "collection",1024)
    for b in range(0, 12, es_bulk_size):
        batch = chunks[b:b + es_bulk_size]
        for doc in batch:
            if "id" in doc:
                del doc["id"]
        doc_store_result = await trio.to_thread.run_sync(lambda: milvus_con.insert(batch, "test", "collection"))
    
    print("insert finish")
    qury="连衣裙有多少"
    m_qury,_=model.encode_queries(qury)
    from doc_store import MatchDenseExpr
    qq=MatchDenseExpr("q_1024_vec",m_qury,"q_1024_vec",distance_type="1")
    # qq.embedding_data=m_qury
    # qq.embedding_data_type="q_1024_vec"
    result=milvus_con.search(matchExprs=[qq],limit=1,indexNames="test",knowledgebaseIds=["collection"])
    print(result)

    
    print(token_count)

if __name__ == "__main__":
    trio.run(main)
