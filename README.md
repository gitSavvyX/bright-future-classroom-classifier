# Bright Future School Classroom Image Classification

An end-to-end image classification pipeline for Bright Future School's classroom photo library, built as **Task 1** of the Ezitech Machine Learning internship.

## Project overview

The client wants to automatically tag its growing media library of classroom photos. This task delivers the **data foundation**: a reproducible, leak-free train/validation split with exploratory analysis proving the split is clean. Model training (Task 2) and deployment (Task 3) build on this foundation in later weeks.

## Dataset

**Student Classroom Behavior (Cropped, Balanced)** — Kaggle  
🔗 https://www.kaggle.com/datasets/abdurasulabduraimov/student-classroom-behavior-cropped-balanced

**7 classes:** `discuss`, `focus`, `hand-raising`, `read`, `standing`, `turn-head`, `write`  
**Total images:** 84,286

> **Substitution note:** The originally suggested `andrewmvd/classroom-images` dataset returned `403 Forbidden` on download (removed from Kaggle). This substitute was chosen because it is ImageFolder-compatible, has 7 clear classroom-activity classes, and supports the required reproducible stratified split.

## Setup

1. Clone the repo and `cd` into it.
2. Create a virtual environment: `python -m venv .venv`
3. Activate it:
   - Windows cmd: `.venv\Scripts\activate.bat`
   - Windows PowerShell: `.venv\Scripts\Activate.ps1`
4. Install dependencies:
