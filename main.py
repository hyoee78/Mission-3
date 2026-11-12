#!/usr/bin/env python3
"""Mini NPU Simulator - E1-3 미션
AI가 계산하는 방식(MAC 연산)을 흉내 내는 작은 계산기

외부 라이브러리(NumPy, pandas 등) 없이 순수 Python 표준 라이브러리만으로 구현되었습니다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

EPSILON = 1e-9


# ============================================================================
# 1. 데이터 구조 (2D Matrix 및 유틸리티)
# ============================================================================

class Matrix2D:
    """N x N 크기의 2차원 패턴 및 필터를 다루는 클래스"""

    def __init__(self, rows: int, cols: int, data: Optional[List[List[float]]] = None):
        if rows <= 0 or cols <= 0:
            raise ValueError(f"유효하지 않은 행렬 크기: {rows}x{cols}")
        self.rows = rows
        self.cols = cols
        if data is None:
            self._data = [[0.0 for _ in range(cols)] for _ in range(rows)]
        else:
            if len(data) != rows or any(len(r) != cols for r in data):
                raise ValueError(f"데이터 크기가 지정된 행렬 크기({rows}x{cols})와 일치하지 않습니다.")
            self._data = [[float(v) for v in row] for row in data]

    def get(self, r: int, c: int) -> float:
        """특정 위치의 값을 조회"""
        self._check_bounds(r, c)
        return self._data[r][c]

    def set(self, r: int, c: int, val: float) -> None:
        """특정 위치의 값을 설정"""
        self._check_bounds(r, c)
        self._data[r][c] = float(val)

    def _check_bounds(self, r: int, c: int) -> None:
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            raise IndexError(f"인덱스 초과: ({r}, {c}), 크기: ({self.rows}, {self.cols})")

    @property
    def data(self) -> List[List[float]]:
        return self._data

    def to_1d(self) -> List[float]:
        """보너스 과제 1: 2차원 배열을 1차원 배열(길이 N^2)로 변환"""
        arr_1d: List[float] = []
        for row in self._data:
            arr_1d.extend(row)
        return arr_1d

    @classmethod
    def from_1d(cls, arr_1d: List[float], rows: int, cols: int) -> "Matrix2D":
        if len(arr_1d) != rows * cols:
            raise ValueError(f"1D 배열 길이({len(arr_1d)})가 {rows}x{cols}={rows*cols}와 일치하지 않습니다.")
        data = [arr_1d[i * cols : (i + 1) * cols] for i in range(rows)]
        return cls(rows, cols, data)


# ============================================================================
# 2. MAC (Multiply-Accumulate) 연산 엔진
# ============================================================================

def mac_2d(mat_a: List[List[float]], mat_b: List[List[float]]) -> float:
    """순수 반복문을 통한 2차원 MAC 연산 구현 (외부 라이브러리 사용 금지)
    
    두 배열을 겹쳐서 같은 위치에 있는 숫자끼리 곱하고 결과를 모두 누적(Accumulate)합니다.
    """
    rows = len(mat_a)
    cols = len(mat_a[0])
    total = 0.0
    for r in range(rows):
        row_a = mat_a[r]
        row_b = mat_b[r]
        for c in range(cols):
            total += row_a[c] * row_b[c]
    return total


def mac_1d(arr_a: List[float], arr_b: List[float]) -> float:
    """보너스 과제 1: 1차원 배열 최적화 MAC 연산 (메모리 순차 접근)"""
    total = 0.0
    length = len(arr_a)
    for i in range(length):
        total += arr_a[i] * arr_b[i]
    return total


# ============================================================================
# 3. 라벨 정규화 및 점수 비교/판정 엔진
# ============================================================================

def normalize_label(raw: Any) -> str:
    """라벨 정규화(표준화): 프로그램 내부 표준 라벨인 'Cross'와 'X'로 변환
    
    규칙:
    - expected 값: '+' -> Cross, 'x' / 'X' -> X
    - filter 키: 'cross' / 'Cross' -> Cross, 'x' / 'X' -> X
    """
    if raw is None:
        return ""
    s = str(raw).strip()
    if s in ("+", "cross", "Cross", "CROSS"):
        return "Cross"
    if s.lower() == "x":
        return "X"
    return s


def decide_mode1(score_a: float, score_b: float, epsilon: float = EPSILON) -> str:
    """모드 1(사용자 입력 3x3) 판정: A / B / 판정 불가"""
    if abs(score_a - score_b) < epsilon:
        return f"판정 불가 (|A-B| < {epsilon:g})"
    if score_a > score_b:
        return "A"
    return "B"


def decide_mode2(score_cross: float, score_x: float, epsilon: float = EPSILON) -> str:
    """모드 2(data.json) 판정: Cross / X / UNDECIDED"""
    if abs(score_cross - score_x) < epsilon:
        return "UNDECIDED"
    if score_cross > score_x:
        return "Cross"
    return "X"


# ============================================================================
# 4. 성능 분석 및 벤치마크 (I/O 제외 순수 연산 시간 측정)
# ============================================================================

def benchmark_mac(mat_a: List[List[float]], mat_b: List[List[float]], iterations: int = 10) -> float:
    """I/O 시간을 제외하고 MAC 연산 함수 호출 구간을 최소 iterations회 반복 측정 후 평균 시간(ms) 반환"""
    start = time.perf_counter()
    for _ in range(iterations):
        _ = mac_2d(mat_a, mat_b)
    end = time.perf_counter()
    avg_ms = ((end - start) / iterations) * 1000.0
    return avg_ms


def benchmark_mac_1d(arr_a: List[float], arr_b: List[float], iterations: int = 10) -> float:
    """1D 배열 MAC 연산 벤치마크 (ms)"""
    start = time.perf_counter()
    for _ in range(iterations):
        _ = mac_1d(arr_a, arr_b)
    end = time.perf_counter()
    avg_ms = ((end - start) / iterations) * 1000.0
    return avg_ms


# ============================================================================
# 5. 패턴 생성기 (보너스 과제 2)
# ============================================================================

class PatternGenerator:
    """크기 N 입력 시, N x N 십자가(Cross)와 X 패턴을 자동 생성"""

    @staticmethod
    def generate_cross(n: int, value: float = 1.0) -> List[List[float]]:
        """N x N 십자가(Cross) 패턴 생성: 중앙 행과 중앙 열을 value로 채움"""
        mat = [[0.0 for _ in range(n)] for _ in range(n)]
        mid = n // 2
        for i in range(n):
            mat[mid][i] = value
            mat[i][mid] = value
        return mat

    @staticmethod
    def generate_x(n: int, value: float = 1.0) -> List[List[float]]:
        """N x N X 패턴 생성: 주 대각선과 부 대각선을 value로 채움"""
        mat = [[0.0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            mat[i][i] = value
            mat[i][n - 1 - i] = value
        return mat


# ============================================================================
# 6. 모드 1: 사용자 입력 (3x3)
# ============================================================================

def read_3x3_matrix(prompt_title: str) -> List[List[float]]:
    """3x3 행렬을 한 줄씩(공백 구분) 입력받아 저장하고 검증하는 함수
    
    행/열 개수 불일치, 숫자 파싱 실패 시 안내 문구를 출력하고 재입력을 유도합니다.
    """
    print(prompt_title)
    while True:
        lines: List[List[float]] = []
        valid = True
        for i in range(3):
            try:
                line = input().strip()
            except (EOFError, KeyboardInterrupt):
                print("\n입력이 중단되었습니다.")
                sys.exit(0)

            tokens = line.split()
            if len(tokens) != 3:
                print("입력 형식 오류: 각 줄에 3개의 숫자를 공백으로 구분해 입력하세요.")
                valid = False
                break

            try:
                row = [float(t) for t in tokens]
                lines.append(row)
            except ValueError:
                print("입력 형식 오류: 각 줄에 3개의 숫자를 공백으로 구분해 입력하세요.")
                valid = False
                break

        if valid and len(lines) == 3:
            return lines
        print("다시 입력해 주세요:")


def run_mode_1(interactive: bool = True, prefill_inputs: Optional[List[List[List[float]]]] = None) -> None:
    """사용자 입력 모드 (3x3 필터 2개와 패턴 1개 입력 후 판정)"""
    print("\n#---------------------------------------")
    print("# [1] 필터 입력")
    print("#---------------------------------------")

    if prefill_inputs:
        filter_a = prefill_inputs[0]
        filter_b = prefill_inputs[1]
        pattern = prefill_inputs[2]
    else:
        filter_a = read_3x3_matrix("필터 A (3줄 입력, 공백 구분)")
        print()
        filter_b = read_3x3_matrix("필터 B (3줄 입력, 공백 구분)")
        print("\n#---------------------------------------")
        print("# [2] 패턴 입력")
        print("#---------------------------------------")
        pattern = read_3x3_matrix("패턴 (3줄 입력, 공백 구분)")

    score_a = mac_2d(pattern, filter_a)
    score_b = mac_2d(pattern, filter_b)
    avg_time_ms = benchmark_mac(pattern, filter_a, iterations=10)
    decision = decide_mode1(score_a, score_b)

    print("\n#---------------------------------------")
    if "판정 불가" in decision:
        print("# [3] MAC 결과 (판정 불가)")
    else:
        print("# [3] MAC 결과")
    print("#---------------------------------------")
    print(f"A 점수: {score_a:.16g}")
    print(f"B 점수: {score_b:.16g}")
    print(f"연산 시간(평균/10회): {avg_time_ms:.4f} ms")
    print(f"판정: {decision}")

    # 3x3 성능 분석
    print("\n#---------------------------------------")
    print("# [4] 성능 분석 (3x3)")
    print("#---------------------------------------")
    print("크기       평균 시간(ms)    연산 횟수")
    print("-------------------------------------")
    print(f"3×3            {avg_time_ms:.4f}           9")


# ============================================================================
# 7. 모드 2: JSON 데이터 분석 (data.json)
# ============================================================================

def load_data_json(filepath: str) -> Dict[str, Any]:
    """data.json 파일 읽기"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"data.json 파일을 찾을 수 없습니다: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def run_mode_2(data_path: str = "data.json") -> Tuple[int, int, int]:
    """data.json 분석 모드 실행"""
    print("\n#---------------------------------------")
    print("# [1] 필터 로드")
    print("#---------------------------------------")

    try:
        data = load_data_json(data_path)
    except Exception as e:
        print(f"데이터 로드 실패: {e}")
        return (0, 0, 0)

    raw_filters: Dict[str, Any] = data.get("filters", {})
    normalized_filters: Dict[str, Dict[str, List[List[float]]]] = {}

    for size_key, filter_pair in raw_filters.items():
        norm_pair: Dict[str, List[List[float]]] = {}
        for f_name, f_mat in filter_pair.items():
            norm_name = normalize_label(f_name)
            norm_pair[norm_name] = f_mat
        normalized_filters[size_key] = norm_pair
        loaded_keys = ", ".join(norm_pair.keys())
        print(f"✓ {size_key:<8} 필터 로드 완료 ({loaded_keys})")

    print("\n#---------------------------------------")
    print("# [2] 패턴 분석 (라벨 정규화 적용)")
    print("#---------------------------------------")

    patterns: Dict[str, Any] = data.get("patterns", {})
    total_tests = len(patterns)
    passed_tests = 0
    failed_tests = 0
    failure_details: List[str] = []

    for pat_key, pat_info in patterns.items():
        print(f"--- {pat_key} ---")
        
        # 키에서 크기(N) 추출: size_{N}_{idx}
        tokens = pat_key.split("_")
        if len(tokens) < 3 or tokens[0] != "size":
            err_msg = f"{pat_key}: 잘못된 패턴 키 형식입니다."
            print(f"FAIL: {err_msg}")
            failed_tests += 1
            failure_details.append(err_msg)
            continue

        size_n = tokens[1]
        target_filter_key = f"size_{size_n}"
        if target_filter_key not in normalized_filters:
            err_msg = f"{pat_key}: 해당 크기의 필터({target_filter_key})가 존재하지 않습니다."
            print(f"FAIL: {err_msg}")
            failed_tests += 1
            failure_details.append(err_msg)
            continue

        target_filter_pair = normalized_filters[target_filter_key]
        if "Cross" not in target_filter_pair or "X" not in target_filter_pair:
            err_msg = f"{pat_key}: 필터 쌍(Cross, X) 구성이 불완전합니다."
            print(f"FAIL: {err_msg}")
            failed_tests += 1
            failure_details.append(err_msg)
            continue

        filter_cross = target_filter_pair["Cross"]
        filter_x = target_filter_pair["X"]
        pattern_input = pat_info.get("input", [])
        expected_raw = pat_info.get("expected")
        expected_norm = normalize_label(expected_raw)

        # 크기 검증: 패턴과 필터 크기 일치 여부
        try:
            p_rows = len(pattern_input)
            p_cols = len(pattern_input[0]) if p_rows > 0 else 0
            f_rows = len(filter_cross)
            f_cols = len(filter_cross[0]) if f_rows > 0 else 0
            if p_rows != f_rows or p_cols != f_cols:
                err_msg = f"{pat_key}: 크기 불일치 (패턴 {p_rows}x{p_cols} != 필터 {f_rows}x{f_cols})"
                print(f"FAIL: {err_msg}")
                failed_tests += 1
                failure_details.append(err_msg)
                continue
        except Exception as e:
            err_msg = f"{pat_key}: 입력 데이터 구조 오류 ({e})"
            print(f"FAIL: {err_msg}")
            failed_tests += 1
            failure_details.append(err_msg)
            continue

        # MAC 연산 수행
        score_cross = mac_2d(pattern_input, filter_cross)
        score_x = mac_2d(pattern_input, filter_x)
        decision = decide_mode2(score_cross, score_x)

        is_pass = (decision == expected_norm)
        status_str = "PASS" if is_pass else "FAIL"

        print(f"Cross 점수: {score_cross:.16g}")
        print(f"X 점수:     {score_x:.16g}")

        if is_pass:
            print(f"판정: {decision} | expected: {expected_norm} | {status_str}")
            passed_tests += 1
        else:
            reason = "동점 규칙" if decision == "UNDECIDED" else "판정 불일치"
            print(f"판정: {decision} | expected: {expected_norm} | {status_str} ({reason})")
            failed_tests += 1
            if decision == "UNDECIDED":
                failure_details.append(f"{pat_key}: 동점(UNDECIDED) 처리 규칙에 따라 FAIL (expected: {expected_norm})")
            else:
                failure_details.append(f"{pat_key}: 판정 {decision} != expected {expected_norm}")

    # 성능 분석 (크기별 MAC 연산 10회 평균 시간 측정)
    print("\n#---------------------------------------")
    print("# [3] 성능 분석 (평균/10회)")
    print("#---------------------------------------")
    print("크기       평균 시간(ms)    연산 횟수")
    print("-------------------------------------")

    # 3x3 임의 패턴으로 측정
    p3 = PatternGenerator.generate_cross(3)
    f3 = PatternGenerator.generate_x(3)
    t3 = benchmark_mac(p3, f3, iterations=10)
    print(f"3×3            {t3:.4f}           9")

    # data.json 필터들을 이용한 5x5, 13x13, 25x25 측정
    for n in (5, 13, 25):
        s_key = f"size_{n}"
        if s_key in normalized_filters:
            fc = normalized_filters[s_key]["Cross"]
            fx = normalized_filters[s_key]["X"]
            t_ms = benchmark_mac(fc, fx, iterations=10)
            op_count = n * n
            print(f"{n}×{n:<2}          {t_ms:.4f}           {op_count}")

    # 결과 요약
    print("\n#---------------------------------------")
    print("# [4] 결과 요약")
    print("#---------------------------------------")
    print(f"총 테스트: {total_tests}개")
    print(f"통과: {passed_tests}개")
    print(f"실패: {failed_tests}개")

    if failure_details:
        print("\n실패 케이스:")
        for det in failure_details:
            print(f"- {det}")

    return (total_tests, passed_tests, failed_tests)


# ============================================================================
# 8. 보너스 과제: 1D 최적화 비교 & 패턴 생성기
# ============================================================================

def run_bonus_tasks() -> None:
    """보너스 과제 실행: 1D 메모리 접근 최적화 비교 및 패턴 생성기 테스트"""
    print("\n==========================================")
    print("=== [보너스 1] 2D vs 1D 최적화 성능 비교 ===")
    print("==========================================")
    print("크기       2D 시간(ms)    1D 시간(ms)    속도 향상(%)")
    print("-------------------------------------------------")

    for n in (3, 5, 13, 25, 50, 100):
        mat_a = PatternGenerator.generate_cross(n)
        mat_b = PatternGenerator.generate_x(n)

        # 1D 변환
        arr_a: List[float] = []
        for row in mat_a:
            arr_a.extend(row)
        arr_b: List[float] = []
        for row in mat_b:
            arr_b.extend(row)

        iterations = 50
        time_2d = benchmark_mac(mat_a, mat_b, iterations=iterations)
        time_1d = benchmark_mac_1d(arr_a, arr_b, iterations=iterations)
        speedup = ((time_2d - time_1d) / time_2d) * 100.0 if time_2d > 0 else 0.0

        print(f"{n}×{n:<3}      {time_2d:.5f}       {time_1d:.5f}       {speedup:+.1f}%")

    print("\n==========================================")
    print("=== [보너스 2] 패턴 생성기(Cross & X) 시연 ===")
    print("==========================================")
    n = 5
    cross_5 = PatternGenerator.generate_cross(n)
    x_5 = PatternGenerator.generate_x(n)

    print(f"생성된 {n}x{n} 십자가(Cross) 패턴:")
    for row in cross_5:
        print(" ", " ".join(f"{int(v)}" for v in row))

    print(f"\n생성된 {n}x{n} X 패턴:")
    for row in x_5:
        print(" ", " ".join(f"{int(v)}" for v in row))


# ============================================================================
# 메인 함수 및 메뉴
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Mini NPU Simulator")
    parser.add_argument("--mode", type=int, choices=[1, 2, 3, 4], help="실행할 모드 번호 (1: 3x3 입력, 2: data.json, 3: 보너스 과제, 4: 전체 실행)")
    parser.add_argument("--data", type=str, default="data.json", help="data.json 경로")
    args = parser.parse_args()

    # 비대화형 플래그가 전달된 경우
    if args.mode == 1:
        run_mode_1()
        return
    elif args.mode == 2:
        run_mode_2(data_path=args.data)
        return
    elif args.mode == 3:
        run_bonus_tasks()
        return
    elif args.mode == 4:
        run_mode_2(data_path=args.data)
        run_bonus_tasks()
        return

    # 대화형 대화 메뉴
    print("=== Mini NPU Simulator ===")
    print("\n[모드 선택]")
    print("1. 사용자 입력 (3x3)")
    print("2. data.json 분석")
    print("3. 보너스 과제 (1D 최적화 비교 & 패턴 생성기)")
    print("4. 종료")

    try:
        choice = input("선택: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n종료합니다.")
        return

    if choice == "1":
        run_mode_1()
    elif choice == "2":
        run_mode_2(data_path=args.data)
    elif choice == "3":
        run_bonus_tasks()
    elif choice == "4":
        print("프로그램을 종료합니다.")
    else:
        print(f"잘못된 선택입니다: {choice}")


if __name__ == "__main__":
    main()
