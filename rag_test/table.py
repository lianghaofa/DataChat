import langchain
from openpyxl import workbook,load_workbook
import numpy as np
import pandas as pd
from xpinyin import Pinyin
import copy
import re
from dateutil.parser import parse as datetime_parse
from rag_tokenizer import tokenizer

def tokenize(d, t, eng):
    d["content_with_weight"] = t
    t = re.sub(r"</?(table|td|caption|tr|th)( [^<>]{0,12})?>", " ", t)
    d["content_ltks"] = tokenizer.tokenize(t)
    d["content_sm_ltks"] = tokenizer.fine_grained_tokenize(d["content_ltks"])

class Excel:

    def __call__(self, file_path, from_page=0,to_page=100000000):
        with open(file_path,"rb") as file:
            wb=Excel._load_excel_to_workbook(file)
        
        total=0
        for sheetname in wb.sheetnames:
            total+=len(list(wb[sheetname].rows))
        print(total)

        res, fails, done = [], [], 0
        rn = 0
        for sheetname in wb.sheetnames:
            ws = wb[sheetname]
            rows = list(ws.rows)
            if not rows:
                continue
            headers = [cell.value for cell in rows[0]]
            missed = set([i for i, h in enumerate(headers) if h is None])
            headers = [
                cell.value for i,
                cell in enumerate(
                    rows[0]) if i not in missed]
            if not headers:
                continue
            data = []
            for i, r in enumerate(rows[1:]):
                rn += 1
                if rn - 1 < from_page:
                    continue
                if rn - 1 >= to_page:
                    break
                row = [
                    cell.value for ii,
                    cell in enumerate(r) if ii not in missed]
                if len(row) != len(headers):
                    fails.append(str(i))
                    continue
                data.append(row)
                done += 1
            if np.array(data).size == 0:
                continue
            res.append(pd.DataFrame(np.array(data), columns=headers))
        print(len(res))
        return res

    @staticmethod
    def _load_excel_to_workbook(file):
        file.seek(0)
        file_head=file.read(4)
        file.seek(0)
        if (file_head.startswith(b'PK\x03\x04') or file_head.startswith(b'\xD0\xCF\x11\xE0')):
            try:
                return load_workbook(file,data_only=True)
            except Exception as e:
                print(f"****wxy: openpyxl load error: {e}, try pandas instead")
        else:    
            print("****wxy: Not an Excel file, converting CSV to Excel Workbook")

def trans_datatime(s):
    try:
        return datetime_parse(s.strip()).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        pass


def trans_bool(s):
    if re.match(r"(true|yes|是|\*|✓|✔|☑|✅|√)$",
                str(s).strip(), flags=re.IGNORECASE):
        return "yes"
    if re.match(r"(false|no|否|⍻|×)$", str(s).strip(), flags=re.IGNORECASE):
        return "no"

def column_data_type(arr):
    arr = list(arr)
    counts = {"int": 0, "float": 0, "text": 0, "datetime": 0, "bool": 0}
    trans = {t: f for f, t in
             [(int, "int"), (float, "float"), (trans_datatime, "datetime"), (trans_bool, "bool"), (str, "text")]}
    for a in arr:
        if a is None:
            continue
        if re.match(r"[+-]?[0-9]{,19}(\.0+)?$", str(a).replace("%%", "")):
            counts["int"] += 1
        elif re.match(r"[+-]?[0-9.]{,19}$", str(a).replace("%%", "")):
            counts["float"] += 1
        elif re.match(r"(true|yes|是|\*|✓|✔|☑|✅|√|false|no|否|⍻|×)$", str(a), flags=re.IGNORECASE):
            counts["bool"] += 1
        elif trans_datatime(str(a)):
            counts["datetime"] += 1
        else:
            counts["text"] += 1
    counts = sorted(counts.items(), key=lambda x: x[1] * -1)
    ty = counts[0][0]
    for i in range(len(arr)):
        if arr[i] is None:
            continue
        try:
            arr[i] = trans[ty](str(arr[i]))
        except Exception:
            arr[i] = None
    # if ty == "text":
    #    if len(arr) > 128 and uni / len(arr) < 0.1:
    #        ty = "keyword"
    return arr, ty

def chunk(filename, from_page=0, to_page=10000000000,
          lang="Chinese", callback=None, **kwargs):
    """
        Excel and csv(txt) format files are supported.
        For csv or txt file, the delimiter between columns is TAB.
        The first line must be column headers.
        Column headers must be meaningful terms inorder to make our NLP model understanding.
        It's good to enumerate some synonyms using slash '/' to separate, and even better to
        enumerate values using brackets like 'gender/sex(male, female)'.
        Here are some examples for headers:
            1. supplier/vendor\tcolor(yellow, red, brown)\tgender/sex(male, female)\tsize(M,L,XL,XXL)
            2. 姓名/名字\t电话/手机/微信\t最高学历（高中，职高，硕士，本科，博士，初中，中技，中专，专科，专升本，MPA，MBA，EMBA）

        Every row in table will be treated as a chunk.
    """

    excel_paser=Excel()
    dfs=excel_paser(filename,from_page,to_page)

    res = []
    PY = Pinyin()
    fieds_map = {
        "text": "_tks",
        "int": "_long",
        "keyword": "_kwd",
        "float": "_flt",
        "datetime": "_dt",
        "bool": "_kwd"}
    for df in dfs:
        for n in ["id", "_id", "index", "idx"]:
            if n in df.columns:
                del df[n]
        clmns = df.columns.values
        txts = list(copy.deepcopy(clmns))
        py_clmns = [
            PY.get_pinyins(
                re.sub(
                    r"(/.*|（[^（）]+?）|\([^()]+?\))",
                    "",
                    str(n)),
                '_')[0] for n in clmns]
        clmn_tys = []
        for j in range(len(clmns)):
            cln, ty = column_data_type(df[clmns[j]])
            clmn_tys.append(ty)
            df[clmns[j]] = cln
            if ty == "text":
                txts.extend([str(c) for c in cln if c])
        clmns_map = [(py_clmns[i].lower() + fieds_map[clmn_tys[i]], str(clmns[i]).replace("_", " "))
                     for i in range(len(clmns))]

        eng = lang.lower() == "english"  # is_english(txts)
        for ii, row in df.iterrows():
            d = {
                "docnm_kwd": filename,
                "title_tks": tokenizer.tokenize(re.sub(r"\.[a-zA-Z]+$", "", filename))
            }
            row_txt = []
            for j in range(len(clmns)):
                if row[clmns[j]] is None:
                    continue
                if not str(row[clmns[j]]):
                    continue
                if not isinstance(row[clmns[j]], pd.Series) and pd.isna(row[clmns[j]]):
                    continue
                fld = clmns_map[j][0]
                d[fld] = row[clmns[j]] if clmn_tys[j] != "text" else tokenizer.tokenize(
                    row[clmns[j]])
                row_txt.append("{}:{}".format(clmns[j], row[clmns[j]]))
            if not row_txt:
                continue
            tokenize(d, "; ".join(row_txt), eng)
            res.append(d)

        # KnowledgebaseService.update_parser_config(
        #     kwargs["kb_id"], {"field_map": {k: v for k, v in clmns_map}})
    # callback(0.35, "")

    return res

    

if __name__ == "__main__":
    
    file_path="./data.xlsx"
    excel=Excel()
    excel(file_path)
    res=chunk(file_path)
    print(len(res))


