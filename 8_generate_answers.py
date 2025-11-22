"""
CDP 답변 업데이트 시스템 (Diff 방식)
- Previous CDP answers (2024년 답변)
- CDP 2025 Updates (질문 변경사항)
- SK Report 증거 검색 (RAG)
- 문장 단위 diff 생성 (유지/수정/추가/삭제)
"""

import json
import os
from typing import Dict, List, Any, Optional
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class CDPAnswerGenerator:
    def __init__(self, config: Dict[str, str]):
        """
        Args:
            config: {
                "previous_answers_path": 전년도 CDP 답변 JSON,
                "updates_path": CDP 2025 업데이트 JSON,
                "qdrant_path": Vector DB 경로,
                "collection_name": Collection 이름
            }
        """
        self.config = config

        # 데이터 로드
        print("📂 데이터 로딩 중...")
        self.previous_answers = self._load_json(config["previous_answers_path"])
        self.updates = self._load_json(config["updates_path"])

        # Vector DB 연결
        print("🔗 Vector DB 연결 중...")
        self.qdrant_client = QdrantClient(path=config["qdrant_path"])
        self.collection_name = config["collection_name"]

        # AI 모델 로드
        print("🧠 Embedding 모델 로딩 중...")
        self.embedding_model = SentenceTransformer('BAAI/bge-m3')

        print("🔀 Reranking 모델 로딩 중...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        print("🤖 OpenAI 클라이언트 초기화 중...")
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        self.generated_answers = {}
        print("✅ 초기화 완료\n")

    def _load_json(self, path: str) -> Dict:
        """JSON 파일 로드"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def generate_all_answers(self, question_ids: List[str]) -> Dict[str, Any]:
        """모든 질문에 대한 답변 생성"""

        print("=" * 80)
        print("CDP 답변 업데이트 생성 시작")
        print("=" * 80)

        for question_id in question_ids:
            print(f"\n{'=' * 80}")
            print(f"질문 {question_id} 처리 중...")
            print(f"{'=' * 80}")

            self.generated_answers[question_id] = self._generate_answer(question_id)

        return self.generated_answers

    def _generate_answer(self, question_id: str) -> Dict[str, Any]:
        """개별 질문에 대한 답변 생성 (Diff 방식)"""

        # 1. Previous answer 가져오기
        previous_data = self.previous_answers.get("questions", {}).get(question_id, {})
        if not previous_data:
            print(f"  ⚠️  이전 답변 없음")
            return {"error": "No previous answer found"}

        question_text = previous_data.get("question_text", "")
        print(f"  📝 질문: {question_text[:80]}...")

        # 2. CDP 2025 Updates 확인
        update_info = self.updates.get("updates", {}).get(question_id)
        cdp_question_updates = self._format_cdp_updates(question_id, update_info)

        if update_info:
            print(f"  📋 CDP 2025 업데이트: {update_info.get('change_type', 'unknown')}")
        else:
            print(f"  📋 CDP 2025 업데이트: 없음 (변경사항 없음)")

        # 3. SK Report에서 증거 검색 (항상 실행)
        print(f"  🔍 Vector DB에서 증거 검색 중...")
        evidence = self._search_evidence(question_text)
        print(f"  ✅ {len(evidence)}개 증거 발견")

        # 4. LLM으로 diff 생성
        print(f"  🤖 LLM으로 답변 업데이트 diff 생성 중...")
        answer_updates = self._generate_diff_with_llm(
            question_id=question_id,
            question_text=question_text,
            previous_data=previous_data,
            evidence=evidence
        )

        # 5. 결과 구성
        result = {
            "question_id": question_id,
            "question_text": question_text,
            "cdp_question_updates": cdp_question_updates,
            "previous_answer_2024": previous_data,
            "sk_2025_report_evidence": evidence[:5],  # 상위 5개
            "suggested_answer_updates": answer_updates,
            "review_flags": self._generate_review_flags(answer_updates, evidence)
        }

        print(f"  ✅ 완료")
        return result

    def _format_cdp_updates(self, question_id: str, update_info: Optional[Dict]) -> Dict:
        """CDP 질문 업데이트 정보 포맷"""

        if not update_info:
            return {
                "has_changes": False,
                "change_type": None,
                "description": "No changes for this question in 2025",
                "scoring_changes": None,
                "impact_on_answer": "You can use the same structure as last year",
                "source": "Corporate_Questionnaires_and_Scoring_Methodologies_Updates_2025_V1.3 (question not listed)"
            }

        change_type = update_info.get("change_type", "unknown")

        # impact 메시지 생성
        impact_messages = {
            "correction": "⚠️ Scoring has changed. Review answer strategy needed.",
            "clarification": "Scoring method unchanged, but description clarified. No structural changes needed.",
            "no_change": "No changes. Same structure as last year.",
            "unknown": "Changes detected. Please review carefully."
        }

        return {
            "has_changes": True,
            "change_type": change_type,
            "description": update_info.get("description", ""),
            "scoring_changes": update_info.get("changes_to_scoring", ""),
            "impact_on_answer": impact_messages.get(change_type, impact_messages["unknown"]),
            "source": f"Corporate_Questionnaires_and_Scoring_Methodologies_Updates_2025_V1.3"
        }

    def _search_evidence(self, question_text: str, top_k: int = 5, initial_k: int = 20) -> List[Dict]:
        """Vector DB에서 관련 증거 검색 (2단계: Vector Search + Reranking)"""

        # 1단계: 벡터 검색
        query_vector = self.embedding_model.encode(question_text).tolist()

        try:
            search_results = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=initial_k
            ).points

            if not search_results:
                return []

            # 2단계: Reranking
            candidates = []
            for result in search_results:
                payload = result.payload if hasattr(result, 'payload') else {}
                text = payload.get("text", payload.get("content", ""))

                candidates.append({
                    "text": text,
                    "page": payload.get("page", payload.get("page_num", 0)),
                    "vector_score": round(result.score, 3),
                    "source": "SK Inc. 2025 Sustainability Report"
                })

            # Cross-encoder로 재점수화
            pairs = [[question_text, cand["text"]] for cand in candidates]
            rerank_scores = self.reranker.predict(pairs)

            for i, cand in enumerate(candidates):
                cand["rerank_score"] = round(float(rerank_scores[i]), 3)
                cand["confidence"] = cand["rerank_score"]

            # 정렬 후 상위 top_k개 반환
            reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
            return reranked

        except Exception as e:
            print(f"  ⚠️  검색 오류: {e}")
            return []

    def _generate_diff_with_llm(
        self,
        question_id: str,
        question_text: str,
        previous_data: Dict,
        evidence: List[Dict]
    ) -> Dict:
        """LLM으로 문장 단위 diff 생성"""

        # 프롬프트 구성
        prompt = self._build_diff_prompt(
            question_id, question_text, previous_data, evidence
        )

        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert CDP (Carbon Disclosure Project) consultant helping SK Inc. update their CDP 2025 submission.

Your task is to analyze previous year's answer and 2025 evidence, then generate sentence-level change suggestions.

CRITICAL OUTPUT FORMAT:
Return a JSON object with this structure:
{
  "changes": [
    {
      "type": "keep",
      "text": "sentence to keep unchanged",
      "reason": "why this is still valid (with evidence page reference)"
    },
    {
      "type": "modify",
      "old_text": "old sentence from 2024",
      "new_text": "updated sentence for 2025",
      "reason": "why this needs to change",
      "evidence_page": 45,
      "evidence_snippet": "relevant quote from 2025 report"
    },
    {
      "type": "add",
      "new_text": "entirely new sentence to add",
      "reason": "why this should be added",
      "evidence_page": 63,
      "evidence_snippet": "supporting quote"
    },
    {
      "type": "delete",
      "old_text": "sentence to remove",
      "reason": "why this is no longer relevant"
    }
  ],
  "final_suggested_answer": {
    // Same JSON structure as previous_answer, but with updated content
  }
}

IMPORTANT:
1. Analyze each sentence/field in the previous answer
2. Compare with 2025 evidence
3. Identify what should be kept, modified, added, or deleted
4. Always cite evidence page numbers
5. Keep the exact same JSON structure as previous answer"""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=4000
            )

            generated_text = response.choices[0].message.content

            # JSON 파싱
            try:
                if "```json" in generated_text:
                    generated_text = generated_text.split("```json")[1].split("```")[0].strip()
                elif "```" in generated_text:
                    generated_text = generated_text.split("```")[1].split("```")[0].strip()

                diff_result = json.loads(generated_text)
                return diff_result

            except Exception as parse_error:
                print(f"  ⚠️  JSON 파싱 실패: {parse_error}")
                return {
                    "changes": [],
                    "final_suggested_answer": previous_data,
                    "parse_error": str(parse_error),
                    "raw_response": generated_text
                }

        except Exception as e:
            print(f"  ⚠️  LLM 생성 오류: {e}")
            return {
                "changes": [],
                "final_suggested_answer": previous_data,
                "error": str(e)
            }

    def _build_diff_prompt(
        self,
        question_id: str,
        question_text: str,
        previous_data: Dict,
        evidence: List[Dict]
    ) -> str:
        """LLM 프롬프트 구성 (Diff 생성용)"""

        # 증거 텍스트 포맷
        evidence_text = "\n\n".join([
            f"Evidence {i+1} (Confidence: {ev['confidence']}, Page: {ev['page']}):\n{ev['text']}"
            for i, ev in enumerate(evidence[:5])
        ]) if evidence else "No evidence found"

        prompt = f"""
# CDP Question {question_id}

## Question Text:
{question_text}

## Previous Year's Answer (2024):
{json.dumps(previous_data, indent=2, ensure_ascii=False)}

## Evidence from SK Inc. 2025 Sustainability Report:
{evidence_text}

## Task:
Compare the 2024 answer with 2025 evidence and generate sentence-level change suggestions.

For each sentence or field in the previous answer:
1. If still valid based on 2025 evidence → type: "keep"
2. If numbers/dates/facts changed → type: "modify" (provide old and new text)
3. If new information should be added → type: "add"
4. If no longer relevant → type: "delete"

**CRITICAL REQUIREMENTS:**
1. Maintain the EXACT same JSON structure as previous answer
2. Cite evidence page numbers for all changes
3. Include evidence snippets for modifications and additions
4. Be specific about what changed (e.g., "10 facilities" → "15 facilities")
5. Keep the same field_name values

**OUTPUT FORMAT:**
Return ONLY valid JSON with "changes" array and "final_suggested_answer" object.
Do NOT include explanations outside the JSON.
"""

        return prompt

    def _generate_review_flags(self, answer_updates: Dict, evidence: List[Dict]) -> Dict:
        """검토 필요 항목 플래그 생성"""

        reasons = []

        # 변경사항 분석
        changes = answer_updates.get("changes", [])

        modify_count = len([c for c in changes if c.get("type") == "modify"])
        add_count = len([c for c in changes if c.get("type") == "add"])
        delete_count = len([c for c in changes if c.get("type") == "delete"])

        if modify_count > 0:
            reasons.append(f"Content modifications ({modify_count} changes)")
        if add_count > 0:
            reasons.append(f"New content added ({add_count} additions)")
        if delete_count > 0:
            reasons.append(f"Content removed ({delete_count} deletions)")
        if len(evidence) < 2:
            reasons.append("Low evidence count (manual verification needed)")

        # Confidence 판단
        if len(evidence) >= 3 and modify_count + add_count + delete_count <= 2:
            confidence = "high"
        elif len(evidence) >= 2:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "needs_review": len(reasons) > 0,
            "reasons": reasons,
            "confidence": confidence,
            "change_summary": {
                "modifications": modify_count,
                "additions": add_count,
                "deletions": delete_count,
                "total_changes": modify_count + add_count + delete_count
            }
        }

    def save_results(self, output_path: str):
        """결과 저장 (영문 버전)"""

        result = {
            "metadata": {
                "year": 2025,
                "company": "SK Inc.",
                "baseline": "2024 CDP Submission",
                "cdp_updates_version": "1.3",
                "evidence_source": "SK Inc. 2025 Sustainability Report",
                "language": "en",
                "output_format": "diff"
            },
            "questions": self.generated_answers
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\n{'=' * 80}")
        print(f"💾 저장 완료: {output_path}")
        print(f"📊 총 {len(self.generated_answers)}개 질문 처리")

        # 통계 출력
        total_changes = 0
        for q_id, answer in self.generated_answers.items():
            changes = answer.get("suggested_answer_updates", {}).get("changes", [])
            total_changes += len(changes)

        print(f"🔄 총 {total_changes}개 변경사항 제안")
        print(f"{'=' * 80}")


if __name__ == "__main__":
    # 설정
    config = {
        "previous_answers_path": "config/previous_cdp_answers.json",
        "updates_path": "config/cdp_2025_updates.json",
        "qdrant_path": "data/qdrant_db",
        "collection_name": "company_docs"
    }

    # 생성기 초기화
    generator = CDPAnswerGenerator(config)

    # 질문 처리 (6개)
    test_questions = ["2.2", "2.2.1", "2.2.2", "2.2.7", "2.3", "2.4"]

    # 답변 생성 (영문)
    results = generator.generate_all_answers(test_questions)

    # 결과 저장 (영문)
    generator.save_results("output/generated_cdp_answers_en.json")
