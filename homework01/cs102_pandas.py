import pandas as pd
import re
from typing import Tuple


# Задача 1
def filter_fsuir_students(data: pd.DataFrame) -> Tuple[int, int, pd.DataFrame]:
    """
    Создает подвыборку студентов факультета систем управления и робототехники (ФСУиР).
    Возвращает количество таких студентов, количество уникальных групп и отфильтрованный датасет.
    """
    fsuir=data[data['факультет']== 'факультет систем управления и робототехники'].copy()
    colvo_fsuir_studients=len(fsuir)
    colvo_fsuir_groups=fsuir['группа'].nunique()
    
    return colvo_fsuir_studients, colvo_fsuir_groups, fsuir

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

    surname_count= df['surname'].value_counts()
    repeated=surname_count[surname_count>1].index
    filter_df=df[df["surname"].isin(repeated)]
    
    filter_df_courses=filter_df.groupby('курс').size()
    filter_df_groupmax=filter_df.groupby('группа').size().idxmax()
    
    return (not filter_df.empty), len(filter_df), filter_df_courses,filter_df_groupmax
    
# Задача 3
def gender_identification(patronym: str) -> str:
    """
    Определяет пол по отчеству. Возвращает пол: female/male/unknown.
    """
    if patronym.endswith(("овна", "евна", "ична", "инична")):
            return "female"
        
    elif patronym.endswith(("ович", "евич", "ич")):
            return "male"
        
    else:
        return "unknown"
    
def analyze_patronyms(df: pd.DataFrame) -> Tuple[int, pd.Series]:
    """
    Определяет количество студентов без отчества и распределение студентов по полу на основе отчества.
    Возвращает:
     - количество студентов без отчества
     - серию с распределением студентов по полу 
    """
    bez_otch=(df["patronim"]=='').sum()
        
    s_otch = df[df["patronim"] != ""]
    
    gender = s_otch["patronim"].str.lower().apply(gender_identification)
    gender_counts = gender.value_counts()
    
    return bez_otch, gender_counts

# Задача 4
def faculty_statistics(data: pd.DataFrame) -> Tuple[pd.DataFrame, Tuple[str, int], Tuple[str, int]]:
    """
    Подсчитывает количество студентов на каждом факультете,
    а также определяет факультеты с максимальным и минимальным числом студентов.
    """
    faculty_count = data['факультет'].value_counts()
    return faculty_count, faculty_count.idxmax(), faculty_count.idxmin()

# Задача 5
def course_statistics(data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Вычисляет среднее и медианное число студентов на каждом курсе.
    Возвращает две серии с результатами: сначала средние, потом медиана.
    """
    counts = data.groupby(["курс", "факультет"]).size()
    
    mean_students = counts.groupby("курс").mean()
    median_students = counts.groupby("курс").median()
    
    return mean_students, median_students

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
    popular_name=data['name'].value_counts().idxmax()
    popular_students=data[data['name']==popular_name]
    popular_group=popular_students['группа'].value_counts().idxmax()
    
    student_info=popular_students[popular_students["группа"]==popular_group].iloc[0]
    name_dolya=round(len(popular_students)/len(data),2)
    return popular_name, popular_group, student_info['факультет'], student_info['курс'], name_dolya

# Задача 7
def find_students_with_name_starting_P(data: pd.DataFrame) -> pd.DataFrame:
    """
    Находит студентов, чье имя встречается ровно один раз и начинается на "П". Выводит их ФИО, факультет и курс.
    """
    names=data['name'].value_counts()
    p_name=names[(names==1) & names.index.str.startswith('П')].index
    
    return data[data["name"].isin(p_name)][["фио", "факультет", "курс"]].copy()
    
# Задача 8
def highest_avg_grade_faculty(data: pd.DataFrame) -> Tuple[str, str, int]:
    """
    Находит факультет, на котором средний балл студентов третьего курса самый высокий.
    Определяет пол, средний балл котого выше.
    Сначала возвращает факультет, затем пол, затем балл.
    """
    third_course=data[data['курс']=='3-й'].copy()
    faculty=third_course.groupby('факультет')['средний_балл'].mean().idxmax()
    faculty_st=third_course[third_course["факультет"]==faculty].copy()
    
    s_otch = faculty_st[faculty_st["patronim"] != ""]
        
    faculty_st['gender']= s_otch["patronim"].str.lower().apply(gender_identification)
    gender_average=faculty_st.groupby('gender')['средний_балл'].mean()
        
    return faculty, gender_average.idxmax(), gender_average.max()


# Задача 9
def find_consecutive_students(data: pd.DataFrame) -> pd.DataFrame:
    """
    Находит первых 5 студентов, которым номера были присвоены подряд.
    Выводит их ФИО, факультет, курс и номер группы.
    """
    data = data.sort_values("ису")

    for i in range(len(data) - 4):
        numbers = data.iloc[i:i + 5]["ису"]

        if (numbers.iloc[1] == numbers.iloc[0] + 1 and
            numbers.iloc[2] == numbers.iloc[1] + 1 and
            numbers.iloc[3] == numbers.iloc[2] + 1 and
            numbers.iloc[4] == numbers.iloc[3] + 1):

            return data.iloc[i:i + 5][
                ["фио", "факультет", "курс", "группа", "ису"]
            ]


if __name__ == "__main__":
    data = pd.read_csv("isu_fake_data.csv")
    data["surname"] = data['фио'].str.split().str[0]
    data["name"] = data['фио'].str.split().str[1]
    data["patronim"] = data['фио'].str.split().str[2].fillna('')
    
    # Задача 1
    num_students, num_groups, fsuir = filter_fsuir_students(data)
    print(f"Студентов на ФСУиР: {num_students}, Групп: {num_groups}")
    
    # Задача 2
    has_homonyms, total_homonyms, homonyms_per_course, max_homonym_group = find_homonymous_students(fsuir)
    print(f"Есть однофамильцы: {has_homonyms}, Всего: {total_homonyms}, Группа с максимумом: {max_homonym_group}")
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
    print(f"Самое популярное имя: {popular_name}, Группа: {name_group}, Факультет: {faculty}, Курс: {course}")
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
