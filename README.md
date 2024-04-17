# FastAPI와 YOLOv5를 활용한 이미지 분석 API

이 프로젝트는 FastAPI를 사용하여 구현된 서버를 통해 이미지를 분석하고 결과를 반환하는 RESTful API입니다. YOLOv5를 이용하여 이미지 내 객체를 식별하고, 해당 정보를 사용자에게 제공합니다.

## 주요 기능

- **이미지 분석**: 사용자로부터 이미지를 받아 YOLOv5를 사용하여 분석합니다.
- **결과 반환**: 분석된 결과를 JSON 형식으로 사용자에게 반환합니다.

## 시작하기

이 섹션에서는 프로젝트를 로컬 환경에서 설정하고 실행하는 방법에 대해 설명합니다.

### 필요 조건

프로젝트를 실행하기 전에 다음 도구들이 설치되어 있어야 합니다:

- Python 3.8 이상
- pip

### 설치 및 실행방법

1. 저장소를 클론합니다:
```bash
git clone https://github.com/chargeFlow12/chargeFlowAPI.git
```
2. 필요한 패키지들을 설치해줍니다.(단 윈도우 환경의 경우 requirements.txt 파일에서 uvloop제거)
```bash
pip install -r requirements.txt
```

3. 서버 실행
```bash
uvicorn main:app --reload
```

### 개발 환경

- FastAPI
- YOLOv5
- Uvicorn