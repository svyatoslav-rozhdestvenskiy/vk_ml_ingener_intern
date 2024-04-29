<h1> Тестовое задание VK </h1>


<p align="left">
  <a href="https://github.com/svyatoslav-rozhdestvenskiy">
    <img alt="Static Badge" src="https://img.shields.io/badge/vk_ml_ingener_intern-%23000000?style=plastic&label=svyatoslav-rozhdestvenskiy&labelColor=%23008000">
  </a>
  <img alt="GitHub top language" src="https://img.shields.io/github/languages/top/svyatoslav-rozhdestvenskiy/vk_ml_ingener_intern?style=plastic&logoColor=008000&labelColor=008000&color=000000">
  <img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/svyatoslav-rozhdestvenskiy/vk_ml_ingener_intern?style=plastic&labelColor=008000&color=000000">



В данном репозитории расположено решение вступительной задачи для стажировки в VK

# Постановка задачи

Имеется датасет для ранжирования intern_task.csv (расположен по [ссылке](https://drive.google.com/file/d/1CcWsfF0gBJSALvc7cjQZFIx63L9XwlnN/view?usp=sharing))

В нем есть **query_id** - айдишник поисковой сессии, фичи релевантности документа по запросу, **rank** - оценка релевантности.

Задача:
1. Подготовить и проверить датасет
2. Натренировать на любом удобном фреймворке модель, которая будет ранжировать документы по их фичам внутри одной сессии
(**query_id**)(по вектору фичей предсказывать ранк документа)
3. Подсчитать метрики ранжирования для своей модели (ndcg_5 как минимум)
4. Оформить решение и выложить на github, gitlab

# Решение

Решение представлено в ноутбуке solution_ml_vk.ipynb

Также имеется файл solution.py в котором представлен код из ноутбука