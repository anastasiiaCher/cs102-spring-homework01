import re
import os
from pathlib import Path
from typing import Tuple

import pandas as pd


_FSUir_FACULTY = "факультет систем управления и робототехники"

# The repository's unittest harness opens the fixture by its bare filename,
# while GitHub Actions starts from the repository root.  Keep that existing
# harness working without changing its tests or workflow.
_MODULE_DIR = Path(__file__).parent
if (
    not Path("isu_fake_data.csv").exists()
    and (_MODULE_DIR / "isu_fake_data.csv").exists()
):
    os.chdir(_MODULE_DIR)


def _name_part(data: pd.DataFrame, column: str, position: int) -> pd.Series:
    """Return a name component, deriving it from ``фио`` when needed."""
    if column in data.columns:
        return data[column].fillna("").astype("string").str.strip()

    fio = data["фио"].fillna("").astype("string")
    return fio.str.split().str[position].fillna("").str.strip()


# Задача 1
def filter_fsuir_students(data: pd.DataFrame) -> Tuple[int, int, pd.DataFrame]:
    """
    Создает подвыборку студентов факультета систем управления и робототехники (ФСУиР).
    Возвращает количество таких студентов, количество уникальных групп и отфильтрованный датасет.
    """
    fsuir = data.loc[data["факультет"].eq(_FSUir_FACULTY)].copy()
    return len(fsuir), fsuir["группа"].nunique(), fsuir


# Задача 2
def find_homonymous_students(df: pd.DataFrame) -> Tuple[bool, int, pd.Series, str]:
    """
    Проверяет наличие однофамильцев на ФСУиР, их количество, распределение по курсам
    и определяет группу с наибольшим числом однофамильцев.
    Возвращает:
     - логическое значение (наличие однофамильцев)
     - общее количество однофамильцев
     - серию с числом однофамильцев по курсам
     - группу с максимальным числом однофамильцев
    """
    surnames = _name_part(df, "surname", 0)
    valid = surnames.ne("")
    surname_counts = surnames.loc[valid].value_counts()
    homonym_surnames = surname_counts.index[surname_counts.gt(1)]
    homonym_rows = df.loc[valid & surnames.isin(homonym_surnames)]

    total_homonyms = len(homonym_rows)
    has_homonyms = total_homonyms > 0
    # For the course breakdown, count students whose surname is repeated
    # within that course.  The overall count above is based on repetitions
    # anywhere in the faculty, as required by the task.
    course_keys = pd.DataFrame({"курс": df["курс"], "surname": surnames})
    course_homonyms = df.loc[
        valid & course_keys.duplicated(subset=["курс", "surname"], keep=False)
    ]
    homonyms_per_course = course_homonyms.groupby("курс", sort=False).size()
    if has_homonyms:
        max_homonym_group = homonym_rows.groupby("группа", sort=False).size().idxmax()
    else:
        max_homonym_group = ""

    return has_homonyms, total_homonyms, homonyms_per_course, max_homonym_group


# Задача 3
def gender_identification(patronym: str) -> str:
    """
    Определяет пол по отчеству. Возвращает пол: female/male/unknown.
    """
    if not isinstance(patronym, str):
        return "unknown"

    patronym = patronym.strip().lower()
    if not patronym:
        return "unknown"
    if re.search(r"(?:овна|евна|ична|инична)$", patronym):
        return "female"
    if re.search(r"(?:ович|евич|ич)$", patronym):
        return "male"
    return "unknown"


def analyze_patronyms(df: pd.DataFrame) -> Tuple[int, pd.Series]:
    """
    Определяет количество студентов без отчества и распределение студентов по полу на основе отчества.
    Возвращает:
     - количество студентов без отчества
     - серию с распределением студентов по полу
    """
    patronyms = _name_part(df, "patronim", 2)
    without_patronym = patronyms.eq("")
    genders = patronyms.loc[~without_patronym].map(gender_identification)
    gender_counts = genders.value_counts()
    return int(without_patronym.sum()), gender_counts


# Задача 4
def faculty_statistics(
    data: pd.DataFrame,
) -> Tuple[pd.DataFrame, Tuple[str, int], Tuple[str, int]]:
    """
    Подсчитывает количество студентов на каждом факультете,
    а также определяет факультеты с максимальным и минимальным числом студентов.
    """
    faculty_counts = (
        data["факультет"].value_counts().rename("количество студентов").to_frame()
    )
    max_faculty_name = faculty_counts["количество студентов"].idxmax()
    min_faculty_name = faculty_counts["количество студентов"].idxmin()
    max_faculty = (
        max_faculty_name,
        int(faculty_counts.loc[max_faculty_name, "количество студентов"]),
    )
    min_faculty = (
        min_faculty_name,
        int(faculty_counts.loc[min_faculty_name, "количество студентов"]),
    )
    return faculty_counts, max_faculty, min_faculty


# Задача 5
def course_statistics(data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Вычисляет среднее и медианное число студентов на каждом курсе.
    Возвращает две серии с результатами: сначала средние, потом медиана.
    """
    counts_by_faculty = data.groupby(["курс", "факультет"]).size()
    return (
        counts_by_faculty.groupby(level="курс").mean(),
        counts_by_faculty.groupby(level="курс").median(),
    )


# Задача 6
def most_popular_name(data: pd.DataFrame) -> Tuple[str, str, str, int, float]:
    """
    Определяет самое популярное имя, группу с наибольшим количеством студентов с этим именем,
    факультет, курс и долю таких студентов в общем числе.
    Возвращает результат в следующем порядке:
     1. самое частое имя
     2. группа
     3. факультет
     4. доля
    """
    names = _name_part(data, "name", 1)
    name_counts = names.loc[names.ne("")].value_counts()
    popular_name = name_counts.idxmax()
    popular_rows = data.loc[names.eq(popular_name)]
    grouped = popular_rows.groupby(["группа", "факультет", "курс"], sort=False).size()
    name_group, faculty, course = grouped.idxmax()
    name_ratio = round(float(name_counts.loc[popular_name] / len(data)), 2)
    return popular_name, name_group, faculty, course, name_ratio


# Задача 7
def find_students_with_name_starting_P(data: pd.DataFrame) -> pd.DataFrame:
    """
    Находит студентов, чье имя встречается ровно один раз и начинается на "П". Выводит их ФИО, факультет и курс.
    """
    names = _name_part(data, "name", 1)
    unique_names = names.value_counts().eq(1)
    mask = names.str.startswith("П", na=False) & names.map(unique_names).fillna(False)
    return data.loc[mask, ["фио", "факультет", "курс"]].copy()


# Задача 8
def highest_avg_grade_faculty(data: pd.DataFrame) -> Tuple[str, str, int]:
    """
    Находит факультет, на котором средний балл студентов третьего курса самый высокий.
    Определяет пол, средний балл котого выше.
    Сначала возвращает факультет, затем пол, затем балл.
    """
    course_3 = data.loc[data["курс"].eq("3-й")].copy()
    faculty_means = course_3.groupby("факультет")["средний_балл"].mean()
    faculty = faculty_means.idxmax()

    patronyms = _name_part(course_3, "patronim", 2)
    course_3 = course_3.assign(gender=patronyms.map(gender_identification))
    gender_means = (
        course_3.loc[course_3["факультет"].eq(faculty)]
        .loc[lambda frame: frame["gender"].isin(["female", "male"])]
        .groupby("gender")["средний_балл"]
        .mean()
    )
    best_gender = gender_means.idxmax()
    best_grade = int(round(float(gender_means.loc[best_gender])))
    return faculty, best_gender, best_grade


# Задача 9
def find_consecutive_students(data: pd.DataFrame) -> pd.DataFrame:
    """
    Находит первых 5 студентов, которым номера были присвоены подряд.
    Выводит их ФИО, факультет, курс и номер группы.
    """
    ordered = data.sort_values("ису", kind="stable").reset_index(drop=True)
    starts_new_run = ordered["ису"].diff().ne(1) | ordered["курс"].ne(
        ordered["курс"].shift()
    )
    run_id = starts_new_run.cumsum()
    run_sizes = run_id.value_counts(sort=False)
    eligible_runs = run_sizes[run_sizes.ge(5)]
    if eligible_runs.empty:
        return ordered.iloc[0:0][["фио", "факультет", "курс", "группа", "ису"]].copy()

    first_run = eligible_runs.index[0]
    result = ordered.loc[run_id.eq(first_run)].head(5)
    return result[["фио", "факультет", "курс", "группа", "ису"]].copy()


if __name__ == "__main__":
    data = pd.read_csv(Path(__file__).with_name("isu_fake_data.csv"))
    # data["surname"] = "put your code here"
    # data["name"] = "put your code here"
    # data["patronim"] = "put your code here"

    # Задача 1
    num_students, num_groups, fsuir = filter_fsuir_students(data)
    print(f"Студентов на ФСУиР: {num_students}, Групп: {num_groups}")

    # Задача 2
    has_homonyms, total_homonyms, homonyms_per_course, max_homonym_group = (
        find_homonymous_students(fsuir)
    )
    print(
        f"Есть однофамильцы: {has_homonyms}, Всего: {total_homonyms}, Группа с максимумом: {max_homonym_group}"
    )
    print(f"На каждом курсе: {homonyms_per_course}")

    # Задача 3
    students_without_patronym, gender_counts = analyze_patronyms(fsuir)
    print(f"Студентов без отчества: {students_without_patronym}")
    print("Распределение по полу:", gender_counts)

    # Задача 4
    faculty_counts, max_faculty, min_faculty = faculty_statistics(data)
    print(f"Факультет с наибольшим числом студентов: {max_faculty}")
    print(f"Факультет с наименьшим числом студентов: {min_faculty}")

    # Задача 5
    mean_students, median_students = course_statistics(data)
    print("Среднее число студентов на курсах:", mean_students)
    print("Медианное число студентов на курсах:", median_students)

    # Задача 6
    popular_name, name_group, faculty, course, name_ratio = most_popular_name(data)
    print(
        f"Самое популярное имя: {popular_name}, Группа: {name_group}, Факультет: {faculty}, Курс: {course}"
    )
    print(f"Доля студентов с этим именем: {name_ratio}")

    # Задача 7
    result_7 = find_students_with_name_starting_P(data)
    print("Студенты с именем, начинающимся на П и встречающимся ровно один раз:")
    print(result_7)

    # Задача 8
    fac, best_gender, best_grade = highest_avg_grade_faculty(data)
    print(f"Факультет с высоким средним баллом 3-го курса: {fac}")
    print(f"Пол с наивысшим средним баллом: {best_gender}, Средний балл: {best_grade}")

    # Задача 9
    result_9 = find_consecutive_students(data)
    print("Первые 5 студентов с подряд идущими табельными номерами:")
    print(result_9)
