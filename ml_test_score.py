import os
import re

TEST_DIR = "tests"

official_categories = {
    "Feature & Data": "test_01_data_integrity.py",
    "Model Development": "test_02_model_development.py",
    "ML Infrastructure": "test_03_ml_infrastructure.py",
    "Monitoring": "test_04_monitoring.py",
    "Mutamorphic Testing": "test_mutamorphic.py",
}

extra_modules = {
    "Preprocessing Module": "test_preprocess.py",
    "Training Module": "test_train.py",
    "Evaluation Module": "test_evaluate.py",
}

def count_tests(file_path):
    if not os.path.exists(file_path):
        return 0
    with open(file_path, "r", encoding="utf-8") as f:
        return len(re.findall(r"def test_", f.read()))

def generate_table(category_map, count_towards_score=True):
    lines = []
    score = 0
    for category, filename in category_map.items():
        path = os.path.join(TEST_DIR, filename)
        test_count = count_tests(path)
        if test_count > 0:
            lines.append(f"| {category:<22} | ✅ {test_count:<8} | ✅         |")
            if count_towards_score:
                score += 2
        else:
            lines.append(f"| {category:<22} | ❌ 0        | ❌         |")
    return lines, score

def main():
    all_lines = []
    all_lines.append("<!-- ML_TEST_SCORE_START -->")
    all_lines.append("| Category              | Test Count | Automated? |")
    all_lines.append("|-----------------------|------------|------------|")

    # Official categories
    official_lines, official_score = generate_table(official_categories)

    # Extra module tests
    extra_lines, _ = generate_table(extra_modules, count_towards_score=False)

    all_lines.extend(official_lines)
    all_lines.extend(extra_lines)
    all_lines.append(f"\n**Final Score:** {min(official_score, 12)}/12")
    all_lines.append("<!-- ML_TEST_SCORE_END -->")

    with open("ml_test_score.md", "w") as f:
        f.write("\n".join(all_lines))

    badge_color = "brightgreen" if official_score >= 10 else "yellow" if official_score >= 6 else "red"
    badge_url = f"https://img.shields.io/badge/ML%20Test%20Score-{min(official_score, 12)}%2F12-{badge_color}"
    with open("ml_test_score_badge.txt", "w") as f:
        f.write(badge_url)

if __name__ == "__main__":
    main()
