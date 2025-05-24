import os
import re

TEST_DIR = "tests"

CATEGORIES = {
    "Feature & Data": "test_01_data_integrity.py",
    "Model Development": "test_02_model_development.py",
    "ML Infrastructure": "test_03_ml_infrastructure.py",
    "Monitoring": "test_04_monitoring.py",
    "Mutamorphic Testing": "test_mutamorphic.py",
    "Preprocessing Module": "test_preprocess.py",
    "Training Module": "test_train.py",
    "Evaluation Module": "test_evaluate.py"
}

def count_tests(file_path):
    if not os.path.exists(file_path):
        return 0
    with open(file_path, "r", encoding="utf-8") as f:
        return len(re.findall(r"def test_", f.read()))

def main():
    total_score = 0
    lines = []
    lines.append("<!-- ML_TEST_SCORE_START -->")
    lines.append("| Category              | Test Count | Automated? |")
    lines.append("|-----------------------|------------|------------|")

    for category, filename in CATEGORIES.items():
        path = os.path.join(TEST_DIR, filename)
        test_count = count_tests(path)
        if test_count > 0:
            lines.append(f"| {category:<22} | ✅ {test_count:<8} | ✅         |")
            total_score += 2
        else:
            lines.append(f"| {category:<22} | ❌ 0        | ❌         |")

    lines.append(f"\n**Final Score:** {total_score}/12")
    lines.append("<!-- ML_TEST_SCORE_END -->")

    with open("ml_test_score.md", "w") as f:
        f.write("\n".join(lines))

    # Optional badge output
    badge_color = "brightgreen" if total_score >= 10 else "yellow" if total_score >= 6 else "red"
    badge_url = f"https://img.shields.io/badge/ML%20Test%20Score-{total_score}%2F12-{badge_color}"
    with open("ml_test_score_badge.txt", "w") as f:
        f.write(badge_url)

if __name__ == "__main__":
    main()
