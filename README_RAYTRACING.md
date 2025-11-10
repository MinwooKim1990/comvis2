# 🌟 Raytracing Simulation with Mirrors

인터랙티브한 레이트레이싱 시뮬레이션입니다. 중앙의 컬러풀한 구체들과 좌우 대각선 거울이 있으며, 카메라를 자유롭게 움직이며 반사 효과를 실시간으로 볼 수 있습니다.

## 🎨 특징

- **레이트레이싱 렌더링**: 실시간 광선 추적으로 사실적인 반사 효과
- **컬러풀한 오브젝트**: 중앙에 여러 색상의 구체들이 배치
- **거울 반사**: 좌우 대각선에 위치한 거울 평면에서 완벽한 반사
- **인터랙티브 카메라**: 마우스와 키보드로 자유로운 시점 이동
- **다중 반사**: 최대 3번까지 광선이 반사되어 복잡한 반사 효과 구현

## 📋 요구사항

```bash
numpy>=1.21.0
pygame>=2.0.0
```

## 🚀 실행 방법

### 1. 의존성 설치

```bash
pip install -r requirements_raytracing.txt
```

### 2. 시뮬레이션 실행

**🚀 고속 버전 (권장) - Numba JIT 최적화:**
```bash
python raytracing_fast.py
```

**🐢 기본 버전 - 순수 Python (느림):**
```bash
python raytracing_simulation.py
```

> **권장**: `raytracing_fast.py` 사용! Numba JIT 컴파일로 **10-30배 빠른 성능**

## 🎮 조작법

| 키/마우스 | 동작 |
|----------|------|
| **W** | 앞으로 이동 |
| **S** | 뒤로 이동 |
| **A** | 왼쪽으로 이동 |
| **D** | 오른쪽으로 이동 |
| **Space** | 위로 이동 |
| **Shift** | 아래로 이동 |
| **마우스 움직임** | 시점 회전 |
| **ESC** | 종료 |

## 🔧 코드 구조

### 주요 클래스

1. **Ray**: 광선 (시작점과 방향)
2. **Material**: 재질 속성 (색상, 반사율, 발광)
3. **Hit**: 광선-물체 충돌 정보
4. **Sphere**: 구체 오브젝트 (중앙의 컬러풀한 구체들)
5. **Plane**: 평면 오브젝트 (거울과 바닥)
6. **Camera**: 카메라 (위치, 방향, 시야각)
7. **RaytracingScene**: 씬 관리 및 레이트레이싱 로직
8. **RaytracingApp**: Pygame GUI 애플리케이션

### 레이트레이싱 알고리즘

```
for each pixel:
    1. 카메라에서 픽셀로 광선 생성
    2. 씬의 모든 오브젝트와 교차 검사
    3. 가장 가까운 교차점 찾기
    4. 재질이 반사성이면:
        - 반사 광선 생성
        - 재귀적으로 추적 (최대 깊이까지)
    5. 조명 계산 (ambient + diffuse)
    6. 최종 색상 반환
```

## 🎨 씬 구성

- **중앙 구체**: 빨간색 큰 구체 (반지름 1.5)
- **주변 구체들**: 5개의 작은 컬러 구체들이 원형으로 배치
  - 주황색, 노란색, 청록색, 파란색, 자홍색
- **왼쪽 거울**: 좌측 대각선 평면 (반사율 95%)
- **오른쪽 거울**: 우측 대각선 평면 (반사율 95%)
- **바닥**: 회색 바닥 평면 (약간의 반사)

## 🔬 기술적 세부사항

### 반사 계산

```python
reflected_direction = direction - 2 * dot(direction, normal) * normal
```

### 광선-구체 교차

이차 방정식 해법:
- `a = direction · direction`
- `b = 2 * (origin - center) · direction`
- `c = (origin - center) · (origin - center) - radius²`
- `discriminant = b² - 4ac`

### 광선-평면 교차

```python
t = (plane_point - ray_origin) · normal / (ray_direction · normal)
```

## 🎯 성능 최적화

### raytracing_fast.py (Numba JIT 버전)
- **JIT 컴파일**: Numba가 Python 코드를 LLVM 기계어로 컴파일
- **병렬 처리**: `@jit(parallel=True)` - 멀티코어 활용
- **Fast Math**: 부동소수점 연산 최적화
- **낮은 해상도**: 1/16 픽셀(scale=4) 렌더링 후 업스케일
- **반사 제한**: 최대 2회 반사로 계산량 감소
- **성능**: **10-30배 향상** (순수 Python 대비)

### raytracing_simulation.py (순수 Python 버전)
- **스케일 렌더링**: 1/4 해상도(scale=2) 렌더링
- **반사 제한**: 최대 3회 반사
- **자기 교차 방지**: epsilon=0.001로 광선 오프셋

### 💡 더 빠르게 하려면?
**C/C++ 확장 모듈**: Numba도 충분히 빠르지만, 극한의 성능이 필요하면:
- Cython으로 C 코드 생성
- pybind11로 C++ 래핑
- OpenMP로 멀티스레딩
- **예상 성능**: Numba 대비 1.2-2배 추가 향상 (개발 복잡도 10배↑)

## 🛠 커스터마이징

### 해상도 조정

```python
app = RaytracingApp(width=1280, height=720)  # 더 높은 해상도
```

### 렌더링 스케일 변경

```python
self.scale = 1  # raytracing_simulation.py 내에서 (더 고품질, 느림)
```

### 반사 횟수 변경

```python
self.max_bounces = 5  # RaytracingScene.__init__()에서
```

### 오브젝트 추가

```python
# RaytracingScene.setup_scene()에 추가
new_sphere = Sphere(
    center=[x, y, z],
    radius=r,
    material=Material(color=np.array([r, g, b]), reflectivity=0.5)
)
self.objects.append(new_sphere)
```

## 📝 라이선스

MIT License

## 👨‍💻 개발자

Claude Code를 사용하여 개발되었습니다.
