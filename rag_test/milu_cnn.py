import logging  
import os  
import re  
import json  
import time  
import copy  
import numpy as np  
import pandas as pd  
from pymilvus import (  
    connections,  
    utility,  
    FieldSchema,  
    CollectionSchema,  
    DataType,  
    Collection,  
    MilvusException  
)  
# from rag import settings  
# from rag.settings import PAGERANK_FLD  
# from rag.utils import singleton  
# from rag_tokenizer import get_project_base_directory  
  
# Import the same interface classes as Infinity  
from doc_store import (  
    DocStoreConnection,  
    MatchExpr,  
    MatchTextExpr,  
    MatchDenseExpr,  
    FusionExpr,  
    OrderByExpr,  
)  
  
logger = logging.getLogger('ragflow.milvus_conn')  

PAGERANK_FLD = "pagerank_fea"

def singleton(cls, *args, **kw):
    instances = {}

    def _singleton():
        key = str(cls) + str(os.getpid())
        if key not in instances:
            instances[key] = cls(*args, **kw)
        return instances[key]

    return _singleton
  
@singleton  
class MilvusConnection(DocStoreConnection):  
    def __init__(self,dbName="",):  
        self.dbName = dbName  
        milvus_uri = "0.0.0.0"
        milvus_port = 19530
        self.username = None
        self.password = None  
          
        # if ":" in milvus_uri:  
        #     host, port = milvus_uri.split(":")  
        #     milvus_port = int(port)  
        # else:  
        #     host = milvus_uri 
        host=milvus_uri 
              
        logger.info(f"Use Milvus {host}:{milvus_port} as the doc engine.")  
          
        # Connect to Milvus  
        for _ in range(24):  
            try:  
                connections.connect(  
                    alias="default",   
                    host=host,   
                    port=milvus_port,  
                    user=self.username if self.username else None,  
                    password=self.password if self.password else None  
                )  
                logger.info(f"Milvus {host}:{milvus_port} is healthy.")  
                break  
            except Exception as e:  
                logger.warning(f"{str(e)}. Waiting Milvus {host}:{milvus_port} to be healthy.")  
                time.sleep(5)  
                  
        if not connections.has_connection("default"):  
            msg = f"Milvus {host}:{milvus_port} is unhealthy in 120s."  
            logger.error(msg)  
            raise Exception(msg)
    
    def dbType(self) -> str:  
        return "milvus"  
  
    def health(self) -> dict:  
        """Return the health status of the database."""  
        try:  
            status = utility.get_server_version()  
            return {  
                "type": "milvus",  
                "status": "green",  
                "version": status  
            }  
        except Exception as e:  
            return {  
                "type": "milvus",  
                "status": "red",  
                "error": str(e)  
            }
    def createIdx(self, indexName: str, knowledgebaseId: str, vectorSize: int):  
        collection_name = f"{indexName}_{knowledgebaseId}"  
        
        from pymilvus import utility
        utility.drop_collection(collection_name)
        # Define fields for the collection  
        fields = [  
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100,auto_id=True),  
            # FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=100),  
            # FieldSchema(name="content_with_weight", dtype=DataType.VARCHAR, max_length=65535),  
            FieldSchema(name="content", dtype=DataType.JSON),  
            FieldSchema(name=f"q_{vectorSize}_vec", dtype=DataType.FLOAT_VECTOR, dim=vectorSize)  
        ]  
        
        # Create collection schema  
        schema = CollectionSchema(fields)  
        
        # Create collection

        collection = Collection(name=collection_name, schema=schema)  
        
        # Create index on vector field  
        index_params = {  
            "metric_type": "COSINE",  
            "index_type": "HNSW",  
            "params": {"M": 16, "efConstruction": 50}  
        }  
        collection.create_index(field_name=f"q_{vectorSize}_vec", index_params=index_params)  
        
        logger.info(f"MILVUS created collection {collection_name}, vector size {vectorSize}")  
    
    def deleteIdx(self, indexName: str, knowledgebaseId: str):  
        collection_name = f"{indexName}_{knowledgebaseId}"  
        try:  
            if utility.has_collection(collection_name):  
                utility.drop_collection(collection_name)  
                logger.info(f"MILVUS dropped collection {collection_name}")  
        except Exception as e:  
            logger.warning(f"MILVUS deleteIdx error: {str(e)}")  
    
    def indexExist(self, indexName: str, knowledgebaseId: str) -> bool:  
        collection_name = f"{indexName}_{knowledgebaseId}"  
        return utility.has_collection(collection_name)

    def insert(self, documents: list[dict], indexName: str, knowledgebaseId: str = None) -> list[str]:  
        collection_name = f"{indexName}_{knowledgebaseId}"  
        
        try:  
            collection = Collection(name=collection_name)  
            collection.load()  
            
            # Prepare data for insertion - only include fields defined in schema
            collection = Collection(name=collection_name)
            schema_fields = [field.name for field in collection.schema.fields]
            
            # Find vector field name
            vector_field = None
            for d in documents:
                for k in d.keys():
                    if k.startswith("q_") and k.endswith("_vec"):
                        vector_field = k
                        break
                if vector_field:
                    break
            
            # Insert documents one by one
            
            for doc in documents:
                vec = doc[vector_field]
                filtered_data = {k: v for k, v in doc.items() if k != vector_field}
                # 将过滤后的字典转换为 JSON 字符串
                content=json.dumps(filtered_data, ensure_ascii=False)
                # if isinstance(vec, np.ndarray):
                #     vec = vec.tolist()  # Convert numpy array to list
                
                # data = {
                #     vector_field: [vec]  # Wrap single vector in a list
                # }
                data={vector_field:vec,
                      "content":content}
                try:
                    collection.insert(data)
                except Exception as e:
                    logger.error(f"MILVUS failed to insert document: {str(e)}")
                    raise
            
            # Flush after all inserts
            collection.flush()
            
            logger.debug(f"MILVUS inserted into {collection_name} {len(documents)} documents.")  
            return []  
        except Exception as e:  
            logger.error(f"MILVUS insert error: {str(e)}")  
            return [str(e)]

    def search(
            self,
            matchExprs: list[MatchExpr],
            limit: int,
            indexNames: str | list[str],
            knowledgebaseIds: list[str]
    ) -> list[dict]:
        """Search for documents matching the given vector expressions."""
        if isinstance(indexNames, str):
            indexNames = indexNames.split(",")
        
        results = []
        
        # Handle vector search
        for indexName in indexNames:
            for knowledgebaseId in knowledgebaseIds:
                collection_name = f"{indexName}_{knowledgebaseId}"
                
                try:
                    if not utility.has_collection(collection_name):
                        continue
                    
                    collection = Collection(name=collection_name)
                    collection.load()
                    
                    # Build the search parameters
                    search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
                    vector_field = None
                    search_vector = None
                    
                    for matchExpr in matchExprs:
                        if isinstance(matchExpr, MatchDenseExpr):
                            vector_field = matchExpr.vector_column_name
                            search_vector = matchExpr.embedding_data
                            break
                    
                    # if not vector_field or not search_vector:
                        # continue
                    
                    # Execute vector search
                    hits = collection.search(
                        data=[search_vector],
                        anns_field=vector_field,
                        param=search_params,
                        limit=limit,
                        output_fields=["content"]
                    )[0]
                    
                    # Process results into simple dict format
                    for hit in hits:
                        content = json.loads(hit.entity.get("content"))
                        results.append({
                            "id": hit.id,
                            "score": hit.score,
                            "content": content
                        })
                
                except Exception as e:
                    logger.error(f"MILVUS search error: {str(e)}")
        
        return results

    def getTotal(self, res: tuple[pd.DataFrame, int] | pd.DataFrame) -> int:  
        if isinstance(res, tuple):  
            return res[1]  
        return len(res)  
    
    def getChunkIds(self, res: tuple[pd.DataFrame, int] | pd.DataFrame) -> list[str]:  
        if isinstance(res, tuple):  
            res = res[0]  
        return list(res["id"])  
    
    def getFields(self, res: tuple[pd.DataFrame, int] | pd.DataFrame, fields: list[str]) -> dict[str, dict]:  
        if isinstance(res, tuple):  
            res = res[0]  
        if not fields:  
            return {}  
        
        fieldsAll = fields.copy()  
        fieldsAll.append('id')  
        column_map = {col.lower(): col for col in res.columns}  
        matched_columns = {column_map[col.lower()]: col for col in set(fieldsAll) if col.lower() in column_map}  
        
        res2 = res[matched_columns.keys()]  
        res2 = res2.rename(columns=matched_columns)  
        res2.drop_duplicates(subset=['id'], inplace=True)  
        
        # Process keyword fields  
        for column in res2.columns:  
            k = column.lower()  
            if self.field_keyword(k):  
                res2[column] = res2[column].apply(lambda v: [kwd for kwd in v.split("###") if kwd])  
        
        return res2.set_index("id").to_dict(orient="index")

    def delete(self, condition: dict, indexName: str, knowledgebaseId: str) -> int:  
        collection_name = f"{indexName}_{knowledgebaseId}"  
        try:  
            # Check if collection exists  
            if not utility.has_collection(collection_name):  
                logger.warning(  
                    f"Skipped deleting from collection {collection_name} since the collection doesn't exist."  
                )  
                return 0  
            
            collection = Collection(collection_name)  
            collection.load()  
            
            # Build Milvus expression from condition  
            expr = self._build_expr_from_condition(condition)  
            
            # Get the ids to be deleted  
            res = collection.query(expr=expr, output_fields=["id"])  
            if not res:  
                return 0  
            
            ids = [doc["id"] for doc in res]  
            
            # Delete documents by ids  
            delete_expr = f"id in {json.dumps(ids)}"  
            deleted_count = collection.delete(delete_expr)  
            collection.release()  
            
            logger.debug(f"MILVUS delete collection {collection_name}, expr {expr}, deleted count {deleted_count}.")  
            return deleted_count  
        except Exception as e:  
            logger.error(f"MILVUS delete error: {str(e)}")  
            return 0

    def get(self, chunkId: str, indexName: str, knowledgebaseIds: list[str]) -> dict | None:  
        try:  
            assert isinstance(knowledgebaseIds, list)  
            for knowledgebaseId in knowledgebaseIds:  
                collection_name = f"{indexName}_{knowledgebaseId}"  
                if not utility.has_collection(collection_name):  
                    logger.warning(  
                        f"Collection not found: {collection_name}, this knowledge base isn't created in Milvus. Maybe it is created in other document engine."  
                    )  
                    continue  
                
                collection = Collection(collection_name)  
                collection.load()  
                
                # Query for the specific chunk ID  
                expr = f"id == '{chunkId}'"  
                results = collection.query(expr=expr, output_fields=["*"])  
                collection.release()  
                
                if results:  
                    # Process the result  
                    result = results[0]  
                    # Convert fields stored as strings back to their original format  
                    for field in result:  
                        if isinstance(result[field], str) and field.endswith('_kwd') and field != "docnm_kwd":  
                            if "###" in result[field]:  
                                result[field] = result[field].split("###")  
                        elif field == "position_int" and isinstance(result[field], str):  
                            arr = [int(hex_val, 16) for hex_val in result[field].split('_')] if result[field] else []  
                            result[field] = [arr[i:i + 5] for i in range(0, len(arr), 5)]  
                        elif field in ["page_num_int", "top_int"] and isinstance(result[field], str):  
                            result[field] = [int(hex_val, 16) for hex_val in result[field].split('_')] if result[field] else []  
                    
                    return result  
            
            return None  
        except Exception as e:  
            logger.error(f"MILVUS get error: {str(e)}")  
            return None
    def getAggregation(self, res: tuple[pd.DataFrame, int] | pd.DataFrame, fieldnm: str):  
        """  
        Milvus doesn't support aggregation operations natively.   
        This implementation provides a workaround by performing   
        the aggregation on the client side.  
        """  
        if isinstance(res, tuple):  
            res = res[0]  
        
        if fieldnm not in res.columns:  
            return list()  
        
        # Get all values for the field  
        values = res[fieldnm].tolist()  
        
        # Flatten the list if field stores arrays  
        flat_values = []  
        for val in values:  
            if isinstance(val, list):  
                flat_values.extend(val)  
            elif isinstance(val, str) and "###" in val:  
                flat_values.extend(val.split("###"))  
            else:  
                flat_values.append(val)  
        
        # Count occurrences of each value  
        from collections import Counter  
        counts = Counter(flat_values)  
        
        # Return as list of tuples (value, count)  
        return list(counts.items())

    def getHighlight(self, res: tuple[pd.DataFrame, int] | pd.DataFrame, keywords: list[str], fieldnm: str):  
        """  
        Milvus doesn't support highlighting natively.  
        This implementation mimics the highlighting behavior seen in InfinityConnection.  
        """  
        if isinstance(res, tuple):  
            res = res[0]  
        
        if fieldnm not in res.columns:  
            return {}  
        
        ans = {}  
        num_rows = len(res)  
        column_id = res["id"]  
        
        for i in range(num_rows):  
            id = column_id[i]  
            txt = res[fieldnm][i]  
            txt = re.sub(r"[\r\n]", " ", txt, flags=re.IGNORECASE | re.MULTILINE)  
            txts = []  
            
            for t in re.split(r"[.?!;\n]", txt):  
                for w in keywords:  
                    t = re.sub(  
                        r"(^|[ .?/'\"\(\)!,:;-])(%s)([ .?/'\"\(\)!,:;-])" % re.escape(w),  
                        r"\1<em>\2</em>\3",  
                        t,  
                        flags=re.IGNORECASE | re.MULTILINE,  
                    )  
                if not re.search(r"<em>[^<>]+</em>", t, flags=re.IGNORECASE | re.MULTILINE):  
                    continue  
                txts.append(t)  
            
            ans[id] = "...".join(txts)  
        
        return ans

    def sql(self, sql: str, fetch_size: int, format: str):  
        """  
        Milvus doesn't support SQL queries natively.  
        This implementation provides a very basic parser for simple SQL queries.  
        For complex queries, a more sophisticated implementation would be needed.  
        """  
        logger.warning("MILVUS sql: Milvus doesn't support SQL natively. Limited functionality is provided.")  
        
        # Very simplified SQL parsing  
        sql = sql.lower()  
        
        if sql.startswith("select"):  
            # Extract collection name and field list  
            from_index = sql.find("from")  
            where_index = sql.find("where")  
            
            if from_index == -1:  
                logger.error("Invalid SQL: FROM clause missing")  
                return None  
            
            fields_part = sql[6:from_index].strip()  
            collection_name = sql[from_index + 4:(where_index if where_index != -1 else len(sql))].strip()  
            
            # Extract fields  
            if fields_part == "*":  
                fields = ["*"]  
            else:  
                fields = [f.strip() for f in fields_part.split(",")]  
            
            # Extract condition if where clause exists  
            expr = ""  
            if where_index != -1:  
                expr = sql[where_index + 5:].strip()  
            
            try:  
                # Execute query  
                if utility.has_collection(collection_name):  
                    collection = Collection(collection_name)  
                    collection.load()  
                    
                    results = collection.query(expr=expr, output_fields=fields if fields != ["*"] else None, limit=fetch_size)  
                    collection.release()  
                    
                    # Format results  
                    if format == "json":  
                        return {"columns": list(results[0].keys()) if results else [], "rows": [list(r.values()) for r in results]}  
                    else:  
                        return results  
                else:  
                    logger.error(f"Collection {collection_name} not found")  
                    return None  
            except Exception as e:  
                logger.error(f"MILVUS sql error: {str(e)}")  
                return None  
        else:  
            logger.error("Only SELECT queries are supported")  
            return None
        
    def update(self, condition: dict, newValue: dict, indexName: str, knowledgebaseId: str) -> bool:  
        collection_name = f"{indexName}_{knowledgebaseId}"  
        try:  
            # Check if collection exists  
            if not utility.has_collection(collection_name):  
                logger.warning(f"Collection {collection_name} doesn't exist")  
                return False  
            
            collection = Collection(collection_name)  
            collection.load()  
            
            # Process special fields in newValue  
            for k, v in list(newValue.items()):  
                if self.field_keyword(k):  
                    if isinstance(v, list):  
                        newValue[k] = "###".join(v)  
                    else:  
                        newValue[k] = v  
                elif re.search(r"_feas$", k):  
                    newValue[k] = json.dumps(v)  
                elif k == 'kb_id':  
                    if isinstance(newValue[k], list):  
                        newValue[k] = newValue[k][0]  # since d[k] is a list, but we need a str  
                elif k == "position_int":  
                    assert isinstance(v, list)  
                    arr = [num for row in v for num in row]  
                    newValue[k] = "_".join(f"{num:08x}" for num in arr)  
                elif k in ["page_num_int", "top_int"]:  
                    assert isinstance(v, list)  
                    newValue[k] = "_".join(f"{num:08x}" for num in v)  
                elif k == "remove":  
                    del newValue[k]  
                    if v in [PAGERANK_FLD]:  
                        newValue[v] = 0  
            
            # If we have a specific ID in the condition, use it for the update  
            if "id" in condition and isinstance(condition["id"], str):  
                # Update specific single document  
                chunk_id = condition["id"]  
                expr = f"id == '{chunk_id}'"  
                
                # Milvus doesn't support partial updates, so we need to get the full document first  
                results = collection.query(expr=expr, output_fields=["*"])  
                if not results:  
                    logger.warning(f"Document with id {chunk_id} not found for update")  
                    collection.release()  
                    return False  
                
                # Update the document with new values  
                doc = results[0]  
                for k, v in newValue.items():  
                    doc[k] = v  
                
                # Delete the old document and insert the updated one  
                collection.delete(expr)  
                collection.insert([doc])  
                
                collection.release()  
                logger.debug(f"MILVUS update collection {collection_name}, id {chunk_id}.")  
                return True  
            else:  
                # Update multiple documents based on condition  
                expr = self._build_expr_from_condition(condition)  
                
                # Get all documents matching the condition  
                results = collection.query(expr=expr, output_fields=["*"])  
                if not results:  
                    logger.warning(f"No documents found matching condition for update")  
                    collection.release()  
                    return False  
                
                # Update each document with new values  
                for doc in results:  
                    for k, v in newValue.items():  
                        doc[k] = v  
                
                # Delete the old documents and insert the updated ones  
                collection.delete(expr)  
                collection.insert(results)  
                
                collection.release()  
                logger.debug(f"MILVUS update collection {collection_name}, condition {expr}, updated {len(results)} documents.")  
                return True  
        except Exception as e:  
            logger.error(f"MILVUS update error: {str(e)}")  
            return False



milvus_con=MilvusConnection()
