import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ---------- Train model silently (no output shown) ----------
df = pd.read_csv(placementdata (1).csv)

df = df[["CGPA", "Internships", "Projects", "Workshops/Certifications",
         "ExtracurricularActivities", "PlacementTraining", "PlacementStatus"]]

for col in ["ExtracurricularActivities", "PlacementTraining"]:
    df[col] = df[col].map({"No": 0, "Yes": 1})

df["PlacementStatus"] = df["PlacementStatus"].map({"NotPlaced": 0, "Placed": 1})

X = df.drop(columns=["PlacementStatus"])
y = df["PlacementStatus"]
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# ---------- Take input from user ----------
print("Enter your details below:\n")

cgpa = float(input("CGPA (e.g. 8.2): "))
internships = int(input("Number of Internships: "))
projects = int(input("Number of Projects: "))
workshops = int(input("Workshops/Certifications completed: "))
extracurricular = input("Extracurricular Activities (Yes/No): ")
training = input("Placement Training done (Yes/No): ")

# ---------- Prepare input & predict ----------
student = pd.DataFrame([{
    "CGPA": cgpa,
    "Internships": internships,
    "Projects": projects,
    "Workshops/Certifications": workshops,
    "ExtracurricularActivities": 1 if extracurricular.strip().lower() == "yes" else 0,
    "PlacementTraining": 1 if training.strip().lower() == "yes" else 0,
}])[feature_names]

student_scaled = scaler.transform(student)

chance = round(
    model.predict_proba(student_scaled)[0][1] * 100,
    2
)

# ---------- Final output ----------
print(f"\nThe chances of getting you placed is {chance} %")

# ---------- Personalized suggestions ----------
suggestions = []

if projects < 3:
    gap = 3 - projects
    suggestions.append(
        f"Add {gap} more Project(s) — placed students typically have 3+ projects, you have {projects}."
    )

if workshops < 2:
    gap = 2 - workshops
    suggestions.append(
        f"Complete {gap} more Workshop/Certification(s) — placed students typically have 2+, you have {workshops}."
    )

if extracurricular.strip().lower() != "yes":
    suggestions.append(
        "Get involved in an Extracurricular Activity — this had a positive relationship with placement in the data."
    )

if training.strip().lower() != "yes":
    suggestions.append(
        "Consider taking Placement Training — students with placement training had a higher placement rate in this dataset."
    )

if internships < 1:
    suggestions.append(
        "Try to get at least 1 Internship, though internships showed a smaller effect compared with some other factors in this dataset."
    )

print("\nSuggestions to improve your chances (ranked by impact):")

if suggestions:
    for i, tip in enumerate(suggestions, 1):
        print(f"{i}. {tip}")
else:
    print("Your profile already covers all the key factors well!")