"""
RAG Evaluation Module
=====================
RAG 파이프라인 성능 평가

평가 지표:
1. Retrieval 평가
   - Hit Rate: 관련 문서가 top-k에 포함되는 비율
   - MRR (Mean Reciprocal Rank): 첫 관련 문서의 순위
   - NDCG: 순위 기반 관련성 점수

2. Generation 평가
   - Faithfulness: 답변이 컨텍스트에 기반하는지
   - Relevance: 답변이 질문에 관련있는지
   - Completeness: 요청된 내용을 모두 포함하는지

3. End-to-End 평가
   - LLM-as-Judge: GPT로 답변 품질 평가

Usage:
    from model.sustainability.evaluator import RAGEvaluator

    evaluator = RAGEvaluator()
    results = evaluator.evaluate_pipeline(rag, test_questions)
    evaluator.save_report(results, "evaluation_report.json")
"""

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re

from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


@dataclass
class RetrievalMetrics:
    """검색 평가 지표"""
    hit_rate: float = 0.0  # 관련 문서 포함 비율
    mrr: float = 0.0  # Mean Reciprocal Rank
    avg_score: float = 0.0  # 평균 유사도 점수
    avg_rerank_score: float = 0.0  # 평균 리랭크 점수


@dataclass
class GenerationMetrics:
    """생성 평가 지표"""
    faithfulness: float = 0.0  # 컨텍스트 충실도 (1-5)
    relevance: float = 0.0  # 질문 관련성 (1-5)
    completeness: float = 0.0  # 완성도 (1-5)
    overall: float = 0.0  # 종합 점수


@dataclass
class QuestionEvaluation:
    """개별 질문 평가 결과"""
    question_id: str
    question_title: str
    retrieval: RetrievalMetrics
    generation: GenerationMetrics
    answer_preview: str
    sources_count: int
    evaluation_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationReport:
    """전체 평가 리포트"""
    timestamp: str
    total_questions: int
    avg_retrieval: RetrievalMetrics
    avg_generation: GenerationMetrics
    questions: List[QuestionEvaluation]
    config: Dict[str, Any] = field(default_factory=dict)


class RAGEvaluator:
    """RAG 파이프라인 평가기"""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        api_key = os.getenv("OPENAI_API_KEY")
        self.openai_client = OpenAI(api_key=api_key) if api_key else None

    def evaluate_retrieval(
        self,
        search_results: List[Any],
        expected_pages: Optional[List[int]] = None,
        expected_sections: Optional[List[str]] = None,
    ) -> RetrievalMetrics:
        """검색 결과 평가"""
        if not search_results:
            return RetrievalMetrics()

        # 기본 점수 계산
        scores = [r.score for r in search_results]
        rerank_scores = [r.rerank_score for r in search_results if r.rerank_score]

        metrics = RetrievalMetrics(
            avg_score=sum(scores) / len(scores) if scores else 0,
            avg_rerank_score=sum(rerank_scores) / len(rerank_scores) if rerank_scores else 0,
        )

        # 예상 페이지가 있으면 Hit Rate 계산
        if expected_pages:
            result_pages = [r.page_num for r in search_results]
            hits = sum(1 for p in expected_pages if p in result_pages)
            metrics.hit_rate = hits / len(expected_pages)

            # MRR 계산
            for i, r in enumerate(search_results):
                if r.page_num in expected_pages:
                    metrics.mrr = 1.0 / (i + 1)
                    break

        return metrics

    def evaluate_generation_with_llm(
        self,
        question: Dict[str, Any],
        answer: str,
        context: str,
    ) -> GenerationMetrics:
        """LLM으로 생성 품질 평가"""
        if not self.openai_client:
            return GenerationMetrics()

        eval_prompt = f"""다음 CDP 질문에 대한 답변을 평가하세요.

## 질문
ID: {question.get('question_id', 'N/A')}
제목: {question.get('title_en', question.get('title', 'N/A'))}

요청 내용:
{json.dumps(question.get('requested_content', []), ensure_ascii=False)}

## 제공된 컨텍스트 (지속가능성 보고서에서 검색됨)
{context[:3000]}...

## 생성된 답변
{answer[:2000]}

---

다음 기준으로 1-5점 척도로 평가하세요:

1. **Faithfulness (충실도)**: 답변이 제공된 컨텍스트에만 기반하는가?
   - 5: 모든 내용이 컨텍스트에서 직접 인용/참조됨
   - 3: 대부분 컨텍스트 기반이나 일부 추론 포함
   - 1: 컨텍스트와 무관한 내용 다수

2. **Relevance (관련성)**: 답변이 질문에 직접적으로 답하는가?
   - 5: 질문의 모든 측면에 직접 답변
   - 3: 질문의 일부에만 답변
   - 1: 질문과 무관한 답변

3. **Completeness (완성도)**: 요청된 내용을 모두 포함하는가?
   - 5: 모든 요청 항목 포함
   - 3: 주요 항목만 포함
   - 1: 대부분 누락

JSON 형식으로 응답하세요:
```json
{{
  "faithfulness": <1-5>,
  "relevance": <1-5>,
  "completeness": <1-5>,
  "reasoning": "평가 근거 한 줄 요약"
}}
```"""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": eval_prompt}],
                temperature=0.1,
                max_tokens=500
            )

            result_text = response.choices[0].message.content

            # JSON 파싱
            json_match = re.search(r'\{[^{}]*\}', result_text, re.DOTALL)
            if json_match:
                scores = json.loads(json_match.group())
                return GenerationMetrics(
                    faithfulness=scores.get("faithfulness", 0),
                    relevance=scores.get("relevance", 0),
                    completeness=scores.get("completeness", 0),
                    overall=(scores.get("faithfulness", 0) +
                            scores.get("relevance", 0) +
                            scores.get("completeness", 0)) / 3
                )
        except Exception as e:
            print(f"LLM evaluation failed: {e}")

        return GenerationMetrics()

    def evaluate_single_question(
        self,
        rag_pipeline,
        question: Dict[str, Any],
        top_k: int = 7,
        use_rerank: bool = True,
        expected_pages: Optional[List[int]] = None,
    ) -> QuestionEvaluation:
        """단일 질문 평가"""
        # 검색 쿼리 구성
        search_query = f"{question.get('title_en', '')} {question.get('rationale', '')}"

        # 검색 수행
        search_results = rag_pipeline.search(
            search_query,
            top_k=top_k,
            use_rerank=use_rerank,
        )

        # 답변 생성
        answer = rag_pipeline.generate_answer(question, search_results)

        # 검색 평가
        retrieval_metrics = self.evaluate_retrieval(
            search_results,
            expected_pages=expected_pages,
        )

        # 컨텍스트 구성
        context = "\n\n".join([
            f"[Page {r.page_num}] {r.content[:500]}"
            for r in search_results[:5]
        ])

        # 생성 평가
        generation_metrics = self.evaluate_generation_with_llm(
            question,
            answer.answer,
            context,
        )

        return QuestionEvaluation(
            question_id=question.get('question_id', 'N/A'),
            question_title=question.get('title_en', question.get('title', 'N/A')),
            retrieval=retrieval_metrics,
            generation=generation_metrics,
            answer_preview=answer.answer[:500] + "..." if len(answer.answer) > 500 else answer.answer,
            sources_count=len(search_results),
            evaluation_details={
                "confidence": answer.confidence,
                "top_source_pages": [r.page_num for r in search_results[:3]],
            }
        )

    def evaluate_pipeline(
        self,
        rag_pipeline,
        questions: List[Dict[str, Any]],
        top_k: int = 7,
        use_rerank: bool = True,
        ground_truth: Optional[Dict[str, List[int]]] = None,
    ) -> EvaluationReport:
        """전체 파이프라인 평가"""
        evaluations = []

        for question in tqdm(questions, desc="Evaluating questions"):
            qid = question.get('question_id', '')
            expected_pages = ground_truth.get(qid) if ground_truth else None

            eval_result = self.evaluate_single_question(
                rag_pipeline,
                question,
                top_k=top_k,
                use_rerank=use_rerank,
                expected_pages=expected_pages,
            )
            evaluations.append(eval_result)

        # 평균 계산
        avg_retrieval = RetrievalMetrics(
            hit_rate=sum(e.retrieval.hit_rate for e in evaluations) / len(evaluations) if evaluations else 0,
            mrr=sum(e.retrieval.mrr for e in evaluations) / len(evaluations) if evaluations else 0,
            avg_score=sum(e.retrieval.avg_score for e in evaluations) / len(evaluations) if evaluations else 0,
            avg_rerank_score=sum(e.retrieval.avg_rerank_score for e in evaluations) / len(evaluations) if evaluations else 0,
        )

        avg_generation = GenerationMetrics(
            faithfulness=sum(e.generation.faithfulness for e in evaluations) / len(evaluations) if evaluations else 0,
            relevance=sum(e.generation.relevance for e in evaluations) / len(evaluations) if evaluations else 0,
            completeness=sum(e.generation.completeness for e in evaluations) / len(evaluations) if evaluations else 0,
            overall=sum(e.generation.overall for e in evaluations) / len(evaluations) if evaluations else 0,
        )

        return EvaluationReport(
            timestamp=datetime.now().isoformat(),
            total_questions=len(evaluations),
            avg_retrieval=avg_retrieval,
            avg_generation=avg_generation,
            questions=evaluations,
            config={
                "top_k": top_k,
                "use_rerank": use_rerank,
                "model": self.model,
            }
        )

    def save_report(
        self,
        report: EvaluationReport,
        output_path: str,
    ) -> Dict[str, Any]:
        """평가 리포트 저장"""
        report_dict = {
            "timestamp": report.timestamp,
            "summary": {
                "total_questions": report.total_questions,
                "retrieval": asdict(report.avg_retrieval),
                "generation": asdict(report.avg_generation),
            },
            "config": report.config,
            "questions": [
                {
                    "question_id": q.question_id,
                    "question_title": q.question_title,
                    "retrieval": asdict(q.retrieval),
                    "generation": asdict(q.generation),
                    "sources_count": q.sources_count,
                    "answer_preview": q.answer_preview,
                    "details": q.evaluation_details,
                }
                for q in report.questions
            ]
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, ensure_ascii=False, indent=2)

        # 요약 출력
        print(f"\n{'='*50}")
        print("RAG Evaluation Report")
        print(f"{'='*50}")
        print(f"Total Questions: {report.total_questions}")
        print(f"\nRetrieval Metrics:")
        print(f"  - Avg Score: {report.avg_retrieval.avg_score:.3f}")
        print(f"  - Avg Rerank Score: {report.avg_retrieval.avg_rerank_score:.3f}")
        if report.avg_retrieval.hit_rate > 0:
            print(f"  - Hit Rate: {report.avg_retrieval.hit_rate:.2%}")
            print(f"  - MRR: {report.avg_retrieval.mrr:.3f}")
        print(f"\nGeneration Metrics (1-5 scale):")
        print(f"  - Faithfulness: {report.avg_generation.faithfulness:.2f}")
        print(f"  - Relevance: {report.avg_generation.relevance:.2f}")
        print(f"  - Completeness: {report.avg_generation.completeness:.2f}")
        print(f"  - Overall: {report.avg_generation.overall:.2f}")
        print(f"\nReport saved to: {output_path}")

        return report_dict

    def compare_configurations(
        self,
        rag_pipeline,
        questions: List[Dict[str, Any]],
        configs: List[Dict[str, Any]],
    ) -> List[EvaluationReport]:
        """여러 설정 비교 평가"""
        results = []

        for config in configs:
            print(f"\n--- Evaluating config: {config} ---")
            report = self.evaluate_pipeline(
                rag_pipeline,
                questions,
                **config
            )
            results.append(report)

        # 비교 요약
        print(f"\n{'='*60}")
        print("Configuration Comparison")
        print(f"{'='*60}")
        print(f"{'Config':<30} {'Retrieval':<12} {'Generation':<12}")
        print("-" * 60)
        for i, (config, report) in enumerate(zip(configs, results)):
            config_name = f"Config {i+1}: rerank={config.get('use_rerank', True)}"
            print(f"{config_name:<30} {report.avg_retrieval.avg_score:.3f}        {report.avg_generation.overall:.2f}")

        return results


def quick_evaluate(
    report_pdf: str,
    questions_json: str,
    output_path: str = "output/rag_evaluation.json",
    sample_size: int = 5,
):
    """빠른 평가 실행"""
    from .rag import EnhancedRAGPipeline
    from .chunker import SustainabilityReportChunker

    print("Step 1: Loading and chunking report...")
    chunker = SustainabilityReportChunker(report_pdf)
    chunks = chunker.parse()

    print("Step 2: Initializing RAG pipeline...")
    rag = EnhancedRAGPipeline()
    rag.create_collection(recreate=True)
    rag.index_chunks(chunks)

    print("Step 3: Loading questions...")
    with open(questions_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    questions = data.get('questions', data)
    if isinstance(questions, dict):
        questions = [questions]

    # 평면화
    def flatten(qs):
        flat = []
        for q in qs:
            flat.append(q)
            if q.get('children'):
                flat.extend(flatten(q['children']))
        return flat

    flat_questions = flatten(questions)[:sample_size]

    print(f"Step 4: Evaluating {len(flat_questions)} questions...")
    evaluator = RAGEvaluator()
    report = evaluator.evaluate_pipeline(rag, flat_questions)
    evaluator.save_report(report, output_path)

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG Pipeline Evaluator")
    parser.add_argument("--report", "-r", required=True, help="Sustainability report PDF")
    parser.add_argument("--questions", "-q", required=True, help="Questions JSON")
    parser.add_argument("--output", "-o", default="output/rag_evaluation.json")
    parser.add_argument("--sample", "-n", type=int, default=5, help="Sample size")

    args = parser.parse_args()

    quick_evaluate(
        args.report,
        args.questions,
        args.output,
        args.sample,
    )
