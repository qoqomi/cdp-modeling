"""
RAG System Accuracy Evaluation Report Generator
================================================
CDP RAG 파이프라인 결과를 분석하고 평가 보고서를 생성합니다.

사용 방법:
    python reports/rag_evaluation_report.py --answers output/cdp_generated_answers.json
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path
import statistics


@dataclass
class RetrievalMetrics:
    """검색 정확도 메트릭"""
    avg_similarity_score: float = 0.0
    min_similarity_score: float = 0.0
    max_similarity_score: float = 0.0
    std_similarity_score: float = 0.0

    # 유사도 분포
    high_confidence_count: int = 0  # >= 0.6
    medium_confidence_count: int = 0  # 0.4 - 0.6
    low_confidence_count: int = 0  # < 0.4

    # 섹션별 분포
    section_distribution: Dict[str, int] = field(default_factory=dict)

    # 페이지 커버리지
    unique_pages_referenced: int = 0
    avg_pages_per_question: float = 0.0


@dataclass
class GenerationMetrics:
    """생성 정확도 메트릭"""
    total_questions: int = 0
    avg_confidence: float = 0.0

    # 신뢰도 분포
    high_confidence_answers: int = 0  # >= 0.7
    medium_confidence_answers: int = 0  # 0.5 - 0.7
    low_confidence_answers: int = 0  # < 0.5

    # 답변 길이 통계
    avg_answer_length: float = 0.0
    min_answer_length: int = 0
    max_answer_length: int = 0


@dataclass
class IngestionMetrics:
    """문서 Ingestion 메트릭"""
    total_chunks_referenced: int = 0
    unique_chunks_used: int = 0
    avg_chunk_length: float = 0.0

    # 소스 다양성
    source_diversity_score: float = 0.0


@dataclass
class QuestionAnalysis:
    """질문별 분석 결과"""
    question_id: str
    question_title: str
    confidence: float
    avg_retrieval_score: float
    num_sources: int
    source_pages: List[int]
    source_sections: List[str]
    answer_length: int
    quality_grade: str  # A, B, C, D


@dataclass
class EvaluationReport:
    """전체 평가 보고서"""
    report_generated_at: str
    data_source: str

    # 요약 통계
    retrieval_metrics: RetrievalMetrics
    generation_metrics: GenerationMetrics
    ingestion_metrics: IngestionMetrics

    # 질문별 상세 분석
    question_analyses: List[QuestionAnalysis]

    # 주요 발견사항
    key_findings: List[str]

    # 개선 권장사항
    improvement_recommendations: List[str]

    # 전체 등급
    overall_grade: str
    overall_score: float


class RAGEvaluator:
    """RAG 시스템 평가기"""

    def __init__(self, answers_path: str):
        self.answers_path = answers_path
        self.data = self._load_data()

    def _load_data(self) -> Dict:
        """데이터 로드"""
        with open(self.answers_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def calculate_retrieval_metrics(self) -> RetrievalMetrics:
        """검색 정확도 메트릭 계산"""
        metrics = RetrievalMetrics()

        all_scores = []
        all_pages = set()
        section_counts = {}
        pages_per_question = []

        for answer in self.data.get('answers', []):
            sources = answer.get('sources', [])
            question_pages = set()

            for source in sources:
                score = source.get('score', 0)
                all_scores.append(score)

                page = source.get('page_num', 0)
                all_pages.add(page)
                question_pages.add(page)

                section = source.get('section') or 'unknown'
                section_counts[section] = section_counts.get(section, 0) + 1

            pages_per_question.append(len(question_pages))

        if all_scores:
            metrics.avg_similarity_score = round(statistics.mean(all_scores), 4)
            metrics.min_similarity_score = round(min(all_scores), 4)
            metrics.max_similarity_score = round(max(all_scores), 4)
            metrics.std_similarity_score = round(statistics.stdev(all_scores) if len(all_scores) > 1 else 0, 4)

            metrics.high_confidence_count = sum(1 for s in all_scores if s >= 0.6)
            metrics.medium_confidence_count = sum(1 for s in all_scores if 0.4 <= s < 0.6)
            metrics.low_confidence_count = sum(1 for s in all_scores if s < 0.4)

        metrics.section_distribution = section_counts
        metrics.unique_pages_referenced = len(all_pages)
        metrics.avg_pages_per_question = round(statistics.mean(pages_per_question), 2) if pages_per_question else 0

        return metrics

    def calculate_generation_metrics(self) -> GenerationMetrics:
        """생성 정확도 메트릭 계산"""
        metrics = GenerationMetrics()

        answers = self.data.get('answers', [])
        metrics.total_questions = len(answers)

        confidences = [a.get('confidence', 0) for a in answers]
        answer_lengths = [len(a.get('answer', '')) for a in answers]

        if confidences:
            metrics.avg_confidence = round(statistics.mean(confidences), 4)
            metrics.high_confidence_answers = sum(1 for c in confidences if c >= 0.7)
            metrics.medium_confidence_answers = sum(1 for c in confidences if 0.5 <= c < 0.7)
            metrics.low_confidence_answers = sum(1 for c in confidences if c < 0.5)

        if answer_lengths:
            metrics.avg_answer_length = round(statistics.mean(answer_lengths), 0)
            metrics.min_answer_length = min(answer_lengths)
            metrics.max_answer_length = max(answer_lengths)

        return metrics

    def calculate_ingestion_metrics(self) -> IngestionMetrics:
        """Ingestion 메트릭 계산"""
        metrics = IngestionMetrics()

        all_chunks = []
        unique_chunks = set()
        chunk_lengths = []

        for answer in self.data.get('answers', []):
            for source in answer.get('sources', []):
                chunk_id = source.get('chunk_id', '')
                all_chunks.append(chunk_id)
                unique_chunks.add(chunk_id)

                preview = source.get('preview', '')
                chunk_lengths.append(len(preview))

        metrics.total_chunks_referenced = len(all_chunks)
        metrics.unique_chunks_used = len(unique_chunks)
        metrics.avg_chunk_length = round(statistics.mean(chunk_lengths), 0) if chunk_lengths else 0

        # 소스 다양성 점수 (unique / total)
        if all_chunks:
            metrics.source_diversity_score = round(len(unique_chunks) / len(all_chunks), 4)

        return metrics

    def analyze_questions(self) -> List[QuestionAnalysis]:
        """질문별 상세 분석"""
        analyses = []

        for answer in self.data.get('answers', []):
            sources = answer.get('sources', [])
            scores = [s.get('score', 0) for s in sources]
            pages = [s.get('page_num', 0) for s in sources]
            sections = [s.get('section') or 'unknown' for s in sources]

            avg_score = statistics.mean(scores) if scores else 0
            confidence = answer.get('confidence', 0)
            answer_length = len(answer.get('answer', ''))

            # 품질 등급 계산
            combined_score = (confidence * 0.6) + (avg_score * 0.4)
            if combined_score >= 0.7:
                grade = 'A'
            elif combined_score >= 0.55:
                grade = 'B'
            elif combined_score >= 0.4:
                grade = 'C'
            else:
                grade = 'D'

            analysis = QuestionAnalysis(
                question_id=answer.get('question_id', 'N/A'),
                question_title=answer.get('question_title', 'N/A')[:50] + '...',
                confidence=confidence,
                avg_retrieval_score=round(avg_score, 4),
                num_sources=len(sources),
                source_pages=pages,
                source_sections=list(set(sections)),
                answer_length=answer_length,
                quality_grade=grade
            )
            analyses.append(analysis)

        return analyses

    def generate_key_findings(
        self,
        retrieval: RetrievalMetrics,
        generation: GenerationMetrics,
        ingestion: IngestionMetrics,
        analyses: List[QuestionAnalysis]
    ) -> List[str]:
        """주요 발견사항 생성"""
        findings = []

        # 전체 성능 평가
        if generation.avg_confidence >= 0.7:
            findings.append(f"[우수] 평균 신뢰도 {generation.avg_confidence:.1%}로 높은 수준의 답변 품질을 보임")
        elif generation.avg_confidence >= 0.5:
            findings.append(f"[보통] 평균 신뢰도 {generation.avg_confidence:.1%}로 중간 수준의 답변 품질")
        else:
            findings.append(f"[개선필요] 평균 신뢰도 {generation.avg_confidence:.1%}로 낮은 수준의 답변 품질")

        # 검색 성능
        if retrieval.avg_similarity_score >= 0.5:
            findings.append(f"[검색] 평균 유사도 {retrieval.avg_similarity_score:.3f}으로 관련 문서를 적절히 검색")
        else:
            findings.append(f"[검색개선필요] 평균 유사도 {retrieval.avg_similarity_score:.3f}으로 검색 품질 개선 필요")

        # 소스 다양성
        findings.append(f"[소스] {ingestion.unique_chunks_used}개의 고유 청크 활용, "
                       f"다양성 점수: {ingestion.source_diversity_score:.2%}")

        # 페이지 커버리지
        findings.append(f"[커버리지] 총 {retrieval.unique_pages_referenced}개 페이지 참조, "
                       f"질문당 평균 {retrieval.avg_pages_per_question}개 페이지")

        # 등급 분포
        grade_counts = {}
        for a in analyses:
            grade_counts[a.quality_grade] = grade_counts.get(a.quality_grade, 0) + 1

        findings.append(f"[등급분포] A: {grade_counts.get('A', 0)}, B: {grade_counts.get('B', 0)}, "
                       f"C: {grade_counts.get('C', 0)}, D: {grade_counts.get('D', 0)}")

        # 섹션 분포
        if retrieval.section_distribution:
            top_sections = sorted(retrieval.section_distribution.items(),
                                key=lambda x: x[1], reverse=True)[:3]
            section_str = ", ".join([f"{s}: {c}회" for s, c in top_sections])
            findings.append(f"[섹션] 주요 참조 섹션 - {section_str}")

        # 낮은 신뢰도 질문 경고
        low_confidence_qs = [a for a in analyses if a.confidence < 0.5]
        if low_confidence_qs:
            q_ids = [a.question_id for a in low_confidence_qs[:5]]
            findings.append(f"[주의] 낮은 신뢰도 질문 ({len(low_confidence_qs)}개): {', '.join(q_ids)}")

        return findings

    def generate_recommendations(
        self,
        retrieval: RetrievalMetrics,
        generation: GenerationMetrics,
        ingestion: IngestionMetrics
    ) -> List[str]:
        """개선 권장사항 생성"""
        recommendations = []

        # 검색 개선 권장
        if retrieval.avg_similarity_score < 0.5:
            recommendations.append(
                "[Retrieval 개선] Hybrid Search(BM25 + Dense Vector) 적용을 권장합니다. "
                "현재 순수 Dense Vector 검색만 사용 중이며, Sparse 검색을 추가하면 "
                "키워드 기반 매칭 정확도가 향상됩니다."
            )
            recommendations.append(
                "[Query Expansion] 질문을 확장하여 검색하는 Query Expansion 기법 적용을 권장합니다. "
                "CDP 질문의 rationale과 ambition을 활용하여 검색 쿼리를 보강할 수 있습니다."
            )

        if retrieval.low_confidence_count > retrieval.high_confidence_count:
            recommendations.append(
                "[Reranking] Cross-Encoder 기반 Reranker 모델 적용을 권장합니다. "
                "초기 검색 결과를 정교하게 재순위화하여 관련성을 높일 수 있습니다."
            )

        # 임베딩 모델 개선
        if retrieval.avg_similarity_score < 0.55:
            recommendations.append(
                "[Embedding Model] 현재 all-MiniLM-L6-v2 모델 대신 더 큰 모델 "
                "(예: e5-large, bge-large) 사용을 고려하세요. 또는 CDP/ESG 도메인에 "
                "fine-tuned된 임베딩 모델 사용을 권장합니다."
            )

        # 청킹 개선
        if ingestion.source_diversity_score < 0.5:
            recommendations.append(
                "[Chunking] 동일한 청크가 반복적으로 사용되고 있습니다. "
                "청크 크기 조정(현재 1000자) 또는 semantic chunking 적용을 고려하세요."
            )

        # 답변 생성 개선
        if generation.low_confidence_answers > 0:
            recommendations.append(
                f"[Generation] {generation.low_confidence_answers}개 질문의 신뢰도가 낮습니다. "
                "해당 질문들에 대해 top_k 값을 늘리거나, 프롬프트 튜닝을 권장합니다."
            )

        # Ground Truth 권장
        recommendations.append(
            "[평가체계] 현재 confidence는 cosine similarity 기반입니다. "
            "정확한 RAG 평가를 위해 Ground Truth 데이터셋 구축 또는 "
            "RAGAS, LLM-as-Judge 방식의 평가 도입을 권장합니다."
        )

        return recommendations

    def calculate_overall_grade(
        self,
        retrieval: RetrievalMetrics,
        generation: GenerationMetrics,
        analyses: List[QuestionAnalysis]
    ) -> tuple:
        """전체 등급 계산"""
        # 가중 점수 계산
        retrieval_score = retrieval.avg_similarity_score * 100
        generation_score = generation.avg_confidence * 100

        # 등급 분포 점수
        grade_points = {'A': 4, 'B': 3, 'C': 2, 'D': 1}
        grade_scores = [grade_points.get(a.quality_grade, 0) for a in analyses]
        avg_grade_score = statistics.mean(grade_scores) if grade_scores else 0

        # 종합 점수 (Retrieval 30%, Generation 40%, Grade Distribution 30%)
        overall_score = (retrieval_score * 0.3) + (generation_score * 0.4) + (avg_grade_score * 25 * 0.3)

        # 등급 결정
        if overall_score >= 85:
            grade = 'A'
        elif overall_score >= 70:
            grade = 'B'
        elif overall_score >= 55:
            grade = 'C'
        else:
            grade = 'D'

        return grade, round(overall_score, 2)

    def generate_report(self) -> EvaluationReport:
        """전체 평가 보고서 생성"""
        retrieval_metrics = self.calculate_retrieval_metrics()
        generation_metrics = self.calculate_generation_metrics()
        ingestion_metrics = self.calculate_ingestion_metrics()
        question_analyses = self.analyze_questions()

        key_findings = self.generate_key_findings(
            retrieval_metrics, generation_metrics, ingestion_metrics, question_analyses
        )

        recommendations = self.generate_recommendations(
            retrieval_metrics, generation_metrics, ingestion_metrics
        )

        overall_grade, overall_score = self.calculate_overall_grade(
            retrieval_metrics, generation_metrics, question_analyses
        )

        return EvaluationReport(
            report_generated_at=datetime.now().isoformat(),
            data_source=os.path.basename(self.answers_path),
            retrieval_metrics=retrieval_metrics,
            generation_metrics=generation_metrics,
            ingestion_metrics=ingestion_metrics,
            question_analyses=question_analyses,
            key_findings=key_findings,
            improvement_recommendations=recommendations,
            overall_grade=overall_grade,
            overall_score=overall_score
        )


def generate_markdown_report(report: EvaluationReport) -> str:
    """마크다운 형식 보고서 생성"""
    md = []

    # 헤더
    md.append("# RAG 시스템 정확도 평가 보고서")
    md.append(f"\n**생성일시**: {report.report_generated_at}")
    md.append(f"**데이터 소스**: {report.data_source}")
    md.append("")

    # 전체 요약
    md.append("---")
    md.append("## 1. 개요 (Executive Summary)")
    md.append("")
    md.append(f"| 항목 | 결과 |")
    md.append(f"|------|------|")
    md.append(f"| **전체 등급** | **{report.overall_grade}** |")
    md.append(f"| **종합 점수** | **{report.overall_score}/100** |")
    md.append(f"| 총 질문 수 | {report.generation_metrics.total_questions} |")
    md.append(f"| 평균 신뢰도 | {report.generation_metrics.avg_confidence:.1%} |")
    md.append(f"| 평균 검색 유사도 | {report.retrieval_metrics.avg_similarity_score:.3f} |")
    md.append("")

    # 주요 발견사항
    md.append("### 주요 발견사항")
    for finding in report.key_findings:
        md.append(f"- {finding}")
    md.append("")

    # Retrieval 정확도
    md.append("---")
    md.append("## 2. Retrieval 정확도 평가 (검색 정확도)")
    md.append("")
    md.append("### 2.1 유사도 점수 통계")
    md.append("| 지표 | 값 |")
    md.append("|------|-----|")
    md.append(f"| 평균 유사도 | {report.retrieval_metrics.avg_similarity_score:.4f} |")
    md.append(f"| 최소 유사도 | {report.retrieval_metrics.min_similarity_score:.4f} |")
    md.append(f"| 최대 유사도 | {report.retrieval_metrics.max_similarity_score:.4f} |")
    md.append(f"| 표준편차 | {report.retrieval_metrics.std_similarity_score:.4f} |")
    md.append("")

    md.append("### 2.2 유사도 분포")
    md.append("| 범위 | 건수 | 비율 |")
    md.append("|------|------|------|")
    total = (report.retrieval_metrics.high_confidence_count +
             report.retrieval_metrics.medium_confidence_count +
             report.retrieval_metrics.low_confidence_count)
    if total > 0:
        md.append(f"| 높음 (≥0.6) | {report.retrieval_metrics.high_confidence_count} | {report.retrieval_metrics.high_confidence_count/total:.1%} |")
        md.append(f"| 중간 (0.4-0.6) | {report.retrieval_metrics.medium_confidence_count} | {report.retrieval_metrics.medium_confidence_count/total:.1%} |")
        md.append(f"| 낮음 (<0.4) | {report.retrieval_metrics.low_confidence_count} | {report.retrieval_metrics.low_confidence_count/total:.1%} |")
    md.append("")

    md.append("### 2.3 섹션별 분포")
    md.append("| 섹션 | 참조 횟수 |")
    md.append("|------|----------|")
    for section, count in sorted(report.retrieval_metrics.section_distribution.items(),
                                  key=lambda x: x[1], reverse=True):
        md.append(f"| {section} | {count} |")
    md.append("")

    md.append("### 2.4 페이지 커버리지")
    md.append(f"- 고유 참조 페이지 수: **{report.retrieval_metrics.unique_pages_referenced}**")
    md.append(f"- 질문당 평균 참조 페이지: **{report.retrieval_metrics.avg_pages_per_question}**")
    md.append("")

    # Document Ingestion
    md.append("---")
    md.append("## 3. Document Ingestion 정확도")
    md.append("")
    md.append("| 지표 | 값 |")
    md.append("|------|-----|")
    md.append(f"| 총 참조 청크 수 | {report.ingestion_metrics.total_chunks_referenced} |")
    md.append(f"| 고유 청크 수 | {report.ingestion_metrics.unique_chunks_used} |")
    md.append(f"| 평균 청크 길이 | {report.ingestion_metrics.avg_chunk_length:.0f} 자 |")
    md.append(f"| 소스 다양성 점수 | {report.ingestion_metrics.source_diversity_score:.1%} |")
    md.append("")

    # Generation 정확도
    md.append("---")
    md.append("## 4. Generation 정확도 평가 (답변 생성)")
    md.append("")
    md.append("### 4.1 신뢰도 분포")
    md.append("| 범위 | 건수 | 비율 |")
    md.append("|------|------|------|")
    total_q = report.generation_metrics.total_questions
    if total_q > 0:
        md.append(f"| 높음 (≥0.7) | {report.generation_metrics.high_confidence_answers} | {report.generation_metrics.high_confidence_answers/total_q:.1%} |")
        md.append(f"| 중간 (0.5-0.7) | {report.generation_metrics.medium_confidence_answers} | {report.generation_metrics.medium_confidence_answers/total_q:.1%} |")
        md.append(f"| 낮음 (<0.5) | {report.generation_metrics.low_confidence_answers} | {report.generation_metrics.low_confidence_answers/total_q:.1%} |")
    md.append("")

    md.append("### 4.2 답변 길이 통계")
    md.append("| 지표 | 값 |")
    md.append("|------|-----|")
    md.append(f"| 평균 답변 길이 | {report.generation_metrics.avg_answer_length:.0f} 자 |")
    md.append(f"| 최소 답변 길이 | {report.generation_metrics.min_answer_length} 자 |")
    md.append(f"| 최대 답변 길이 | {report.generation_metrics.max_answer_length} 자 |")
    md.append("")

    # 질문별 상세 분석
    md.append("---")
    md.append("## 5. 질문별 상세 분석")
    md.append("")
    md.append("| Question ID | 신뢰도 | 검색점수 | 소스수 | 등급 |")
    md.append("|-------------|--------|----------|--------|------|")
    for qa in report.question_analyses:
        md.append(f"| {qa.question_id} | {qa.confidence:.1%} | {qa.avg_retrieval_score:.3f} | {qa.num_sources} | {qa.quality_grade} |")
    md.append("")

    # 등급 분포 요약
    grade_counts = {}
    for qa in report.question_analyses:
        grade_counts[qa.quality_grade] = grade_counts.get(qa.quality_grade, 0) + 1

    md.append("### 등급 분포 요약")
    md.append("| 등급 | 건수 | 비율 |")
    md.append("|------|------|------|")
    for grade in ['A', 'B', 'C', 'D']:
        count = grade_counts.get(grade, 0)
        ratio = count / len(report.question_analyses) if report.question_analyses else 0
        md.append(f"| {grade} | {count} | {ratio:.1%} |")
    md.append("")

    # 개선 권장사항
    md.append("---")
    md.append("## 6. 개선 권장사항")
    md.append("")
    for i, rec in enumerate(report.improvement_recommendations, 1):
        md.append(f"### {i}. {rec.split(']')[0]}]")
        md.append(f"{rec.split(']')[1].strip()}")
        md.append("")

    # 결론
    md.append("---")
    md.append("## 7. 결론")
    md.append("")
    md.append(f"### 전체 RAG 시스템 평가 결과: **{report.overall_grade}등급** ({report.overall_score}/100점)")
    md.append("")

    if report.overall_grade == 'A':
        md.append("시스템이 우수한 성능을 보이고 있습니다. 현재 수준을 유지하면서 지속적인 모니터링을 권장합니다.")
    elif report.overall_grade == 'B':
        md.append("시스템이 양호한 성능을 보이고 있으나, 일부 개선이 필요합니다. 위의 권장사항을 검토하여 성능 향상을 고려하세요.")
    elif report.overall_grade == 'C':
        md.append("시스템 성능이 보통 수준입니다. 검색 및 생성 품질 개선을 위한 적극적인 조치가 필요합니다.")
    else:
        md.append("시스템 성능이 미흡합니다. 위의 개선 권장사항을 우선적으로 적용하여 품질을 향상시켜야 합니다.")

    md.append("")
    md.append("---")
    md.append(f"*이 보고서는 {report.report_generated_at}에 자동 생성되었습니다.*")

    return "\n".join(md)


def generate_json_report(report: EvaluationReport) -> Dict:
    """JSON 형식 보고서 생성"""
    return {
        "report_generated_at": report.report_generated_at,
        "data_source": report.data_source,
        "overall_grade": report.overall_grade,
        "overall_score": report.overall_score,
        "retrieval_metrics": asdict(report.retrieval_metrics),
        "generation_metrics": asdict(report.generation_metrics),
        "ingestion_metrics": asdict(report.ingestion_metrics),
        "question_analyses": [asdict(qa) for qa in report.question_analyses],
        "key_findings": report.key_findings,
        "improvement_recommendations": report.improvement_recommendations
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RAG Evaluation Report Generator")
    parser.add_argument("--answers", "-a", required=True, help="Generated answers JSON path")
    parser.add_argument("--output-dir", "-o", default="output/reports/", help="Output directory")
    parser.add_argument("--format", "-f", choices=['md', 'json', 'both'], default='both',
                       help="Output format")

    args = parser.parse_args()

    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 평가 실행
    print(f"\n{'='*50}")
    print("RAG System Evaluation Report Generator")
    print(f"{'='*50}")

    evaluator = RAGEvaluator(args.answers)
    report = evaluator.generate_report()

    # 결과 출력
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.format in ['md', 'both']:
        md_content = generate_markdown_report(report)
        md_path = output_dir / f"rag_evaluation_report_{timestamp}.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"Markdown report saved to: {md_path}")

    if args.format in ['json', 'both']:
        json_content = generate_json_report(report)
        json_path = output_dir / f"rag_evaluation_report_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_content, f, ensure_ascii=False, indent=2)
        print(f"JSON report saved to: {json_path}")

    # 요약 출력
    print(f"\n{'='*50}")
    print("EVALUATION SUMMARY")
    print(f"{'='*50}")
    print(f"Overall Grade: {report.overall_grade}")
    print(f"Overall Score: {report.overall_score}/100")
    print(f"Total Questions: {report.generation_metrics.total_questions}")
    print(f"Average Confidence: {report.generation_metrics.avg_confidence:.1%}")
    print(f"Average Retrieval Score: {report.retrieval_metrics.avg_similarity_score:.3f}")
    print(f"\nKey Findings:")
    for finding in report.key_findings[:5]:
        print(f"  - {finding}")


if __name__ == "__main__":
    main()
