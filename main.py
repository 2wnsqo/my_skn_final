from fastapi import FastAPI, HTTPException
from api_models import QuestionRequest, QuestionResponse, PlansRequest, PlansResponse
from api_service import InterviewEvaluationService
import uvicorn

# FastAPI 앱 초기화
app = FastAPI(
    title="면접 평가 API",
    description="실시간 면접 평가 시스템 - 개별 질문 평가 및 전체 면접 평가",
    version="1.0.0"
)

# 전역 서비스 인스턴스
evaluation_service = None

@app.on_event("startup")
async def startup_event():
    """서버 시작시 서비스 초기화"""
    global evaluation_service
    print("🚀 면접 평가 API 서버 시작 중...")
    evaluation_service = InterviewEvaluationService()
    print("✅ 서비스 초기화 완료!")

@app.get("/")
async def root():
    """API 상태 확인"""
    return {
        "message": "면접 평가 API 서버가 정상 작동 중입니다.",
        "version": "1.0.0",
        "status": "healthy",
        "service_initialized": evaluation_service is not None,
        "endpoints": [
            "/interview/evaluate/feedback",
            "/interview/evaluate/plans"
        ]
    }

@app.post("/interview/evaluate/feedback", response_model=QuestionResponse)
async def evaluate_feedback(request: QuestionRequest):
    """
    전체 면접 질문-답변 일괄 평가 엔드포인트 (자동 최종 평가 포함)
    
    Args:
        request: 사용자 정보 + 질문-답변 쌍들
        
    Returns:
        QuestionResponse: 전체 평가 결과 (개별 + 최종 + 계획)
    """
    try:
        if not evaluation_service:
            raise HTTPException(status_code=500, detail="서비스가 초기화되지 않았습니다.")
        
        # 전체 질문 일괄 평가 + 자동 최종 평가 실행
        result = evaluation_service.evaluate_multiple_questions(
            user_id=request.user_id,
            qa_pairs=request.qa_pairs,
            ai_resume_id=request.ai_resume_id,
            user_resume_id=request.user_resume_id,
            posting_id=request.posting_id,
            company_id=request.company_id,
            position_id=request.position_id
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["message"])
        
        return QuestionResponse(
            success=True,
            interview_id=result["interview_id"],
            message=result["message"],
            total_questions=result["total_questions"],
            overall_score=result.get("overall_score"),
            overall_feedback=result.get("overall_feedback"),
            per_question_results=result.get("per_question_results"),
            interview_plan=result.get("interview_plan")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")

@app.post("/interview/evaluate/plans", response_model=PlansResponse)
async def evaluate_plans(request: PlansRequest):
    """
    면접 준비 계획 생성 엔드포인트
    
    Args:
        request: 면접 ID
        
    Returns:
        PlansResponse: 면접 준비 계획 결과
    """
    try:
        if not evaluation_service:
            raise HTTPException(status_code=500, detail="서비스가 초기화되지 않았습니다.")
        
        # 면접 준비 계획 생성
        result = evaluation_service.generate_interview_plans(
            interview_id=request.interview_id
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["message"])
        
        return PlansResponse(
            success=True,
            interview_plan=result["interview_plan"],
            plan_id=result.get("plan_id"),
            message=result["message"],
            interview_id=result["interview_id"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")


if __name__ == "__main__":
    print("면접 평가 API 서버를 시작합니다...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )