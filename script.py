# Importation des bibliothèques nécessaires
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import matplotlib.pyplot as plt


# Création de la session Spark
spark = SparkSession.builder \
    .appName("Exercice Machine Learning avec PySpark - Higgs Dataset") \
    .getOrCreate()

# Chargement de l'ensemble de données Higgs
higgs_df = spark.read.option("header", "false").option("inferSchema", "true").csv("HIGGS.csv")

# Renommage de la colonne cible
higgs_df = higgs_df.withColumnRenamed("_c0", "label")

# Suppression des enregistrements avec au moins une valeur nulle dans une colonne
higgs_df = higgs_df.dropna()

# Sélection aléatoire d'un sous-ensemble des données pour accélérer l'exécution des calculs dans cet exemple
# higgs_df = higgs_df.sample(False, 0.01, seed=42)

# Suréchantillonnage ou sous-échantillonnage pour équilibrer les classes
# Ici, nous allons utiliser la méthode de sous-échantillonnage en réduisant le nombre d'échantillons de la classe majoritaire
major_class_count = higgs_df.filter(higgs_df["label"] == 0).count()
minor_class_count = higgs_df.filter(higgs_df["label"] == 1).count()

if major_class_count > minor_class_count:
    sampled_major_class = higgs_df.filter(higgs_df["label"] == 0).sample(False, minor_class_count / major_class_count)
    sampled_minor_class = higgs_df.filter(higgs_df["label"] == 1)
else:
    sampled_major_class = higgs_df.filter(higgs_df["label"] == 0)
    sampled_minor_class = higgs_df.filter(higgs_df["label"] == 1).sample(False, major_class_count / minor_class_count)

higgs_df = sampled_major_class.union(sampled_minor_class)

# Exploration rapide du dataset
print("Nombre total de lignes dans l'ensemble de données Higgs:", higgs_df.count())

# Visualisation des classes
class_counts = higgs_df.groupBy("label").count().collect()
class_labels = [row["label"] for row in class_counts]
class_counts = [row["count"] for row in class_counts]
plt.bar(class_labels, class_counts, color=['blue', 'red'])
plt.xlabel('Classe')
plt.ylabel('Nombre d\'événements')
plt.title('Distribution des classes')
plt.show()

# Feature Engineering
assembler = VectorAssembler(inputCols=higgs_df.columns[1:], outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

# Division du dataset en ensemble d'entraînement et ensemble de test
(train_data, test_data) = higgs_df.randomSplit([0.8, 0.2], seed=42)

# Initialisation des modèles
lr = LogisticRegression(featuresCol="scaled_features", labelCol="label")
rf = RandomForestClassifier(featuresCol="scaled_features", labelCol="label")
gbt = GBTClassifier(featuresCol="scaled_features", labelCol="label")

# Création du pipeline
pipeline_lr = Pipeline(stages=[assembler, scaler, lr])
pipeline_rf = Pipeline(stages=[assembler, scaler, rf])
pipeline_gbt = Pipeline(stages=[assembler, scaler, gbt])

# Entraînement des modèles
model_lr = pipeline_lr.fit(train_data)
model_rf = pipeline_rf.fit(train_data)
model_gbt = pipeline_gbt.fit(train_data)

# Prédictions sur l'ensemble de test
predictions_lr = model_lr.transform(test_data)
predictions_rf = model_rf.transform(test_data)
predictions_gbt = model_gbt.transform(test_data)

# Évaluation des performances
evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")

# incicateur auc = air sur la courbe de roc
auc_lr = evaluator.evaluate(predictions_lr)
auc_rf = evaluator.evaluate(predictions_rf)
auc_gbt = evaluator.evaluate(predictions_gbt)

print("AUC pour la régression logistique:", auc_lr)
print("AUC pour Random Forest:", auc_rf)
print("AUC pour Gradient Boosted Trees:", auc_gbt)

# Génération de la visualisation des données
plt.figure(figsize=(10, 6))
plt.bar(['Régression Logistique', 'Random Forest', 'GBT'], [auc_lr, auc_rf, auc_gbt], color='r')
plt.title('Comparaison des performances des modèles')
plt.xlabel('Modèles')
plt.ylabel('AUC')
plt.grid(True)
plt.show() 


# Arrêt de la session Spark
spark.stop()
