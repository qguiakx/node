from langchain_core.retrievers import BaseRetriever
from typing import List
from langchain_core.documents import Document


class RRFRetriever(BaseRetriever):
    retrievers: list
    k: int = 60

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # 获取所有检索器的结果
        all_results = []
        for i, retriever in enumerate(self.retrievers):
            results = retriever.invoke(query)
            all_results.extend([(doc, i, rank) for rank, doc in enumerate(results, 1)])

        # 计算RRF分数
        doc_scores = {}
        for doc, retriever_idx, rank in all_results:
            doc_id = str(hash(doc.page_content))
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {"doc": doc, "score": 0}
            # RRF公式: score += 1/(rank + k)
            doc_scores[doc_id]["score"] += 1 / (rank + self.k)

        # 按分数排序
        sorted_items = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)

        # 计算最大分数以归一化
        if sorted_items:
            max_score = sorted_items[0]["score"]
            # 过滤相关性高于95%的文档，最多返回3个
            relevant_docs = []
            for item in sorted_items:
                # 归一化分数到0-100%范围
                normalized_score = (item["score"] / max_score) * 100
                if normalized_score >= 97:
                    relevant_docs.append(item["doc"])
                    if len(relevant_docs) >= 3:
                        break
            return relevant_docs
        return []