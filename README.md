# Wine Quality Predictor

This project demonstrates a machine learning pipeline for predicting wine quality based on various chemical features. The pipeline includes data preprocessing, model selection (Random Forest, K-Nearest Neighbors, SVM), hyperparameter tuning, and deployment as an API using Flask.

## Project Structure

### 1. API (Flask Server)
The `app.py` contains the backend Flask server responsible for serving a machine learning model that predicts the wine quality based on chemical features. The model and scaler are loaded from `.pkl` files and used to make predictions.

#### Key Files:
- `app.py`: Defines the Flask API with the `/predict` route for predicting wine quality based on features.
- `best_model.pkl`: Contains the best-trained model.
- `scaler.pkl`: Contains the scaler used to preprocess the features.

#### How to Run:
```bash
# Install dependencies
pip install -r requirements.txt
```

#### Run the Flask server

```
python app.py
```

#### API Endpoint:

```
POST /predict:
```


##### Request Body:
```
{
  "features": [7.88, 1.44, 0.81, 9.81, 0.47, 29, 44, 0.99208, 2.83, 1.31, 8.55]
}
```

##### Response:

```
{
  "prediction": 5
}
```

### 2. Model Training
The train_model.py script contains the machine learning pipeline for training the model on the Wine Quality dataset, tuning hyperparameters, and saving the best model and scaler for production use.

##### Key Steps:
Data Preprocessing: Splitting data into training and testing sets.
Model Pipelines: Defining Random Forest, KNN, and SVM pipelines with normalization and standardization.
GridSearchCV: Performing grid search on hyperparameters to find the best model.
Model Saving: Saving the best model and scaler using joblib.
How to Run:

##### Install dependencies

```
pip install -r requirements.txt
```

##### Train the model

```
python train_model.py
```

### 3. Frontend (HTML)
The index.html provides a simple user interface to input wine features and make predictions using the Flask API. The form allows users to either manually input values or generate random values based on typical wine chemical properties.

#### Features:
Input fields for all 11 chemical features.
A button to generate random valid values for each feature.
Display the predicted wine quality based on the model's output.

#### How to Use:

Open index.html in a web browser.
Fill out the form or generate random values.
Submit the form to predict the wine quality.

### 4. Unit Tests (Pytest)

Unit tests are defined to ensure the model and preprocessing steps function correctly.

#### Tests Included:

Model Loading: Verifies that the model and scaler are loaded correctly.
Model Performance: Ensures the model accuracy is above a certain threshold on a test dataset.
Scaler Behavior: Ensures the scaler is applied correctly during preprocessing.

#### How to Run:

```
# Install pytest
pip install pytest
```

#### Run the tests

pytest
Installation
Clone the current repository:

Install the required Python libraries:

```
pip install -r requirements.txt
```

Download the dataset or use the existing one from UCI:

```
python train_model.py
```

## Technologies Used
Flask: For serving the machine learning model via an API.
Scikit-learn: For training and testing machine learning models.
Joblib: For saving and loading models.
Pytest: For unit testing.
HTML/CSS: For creating a simple frontend.

## License
This project is licensed under the MIT License.

___________________________________________________________________________________________________________________________________

# Previsão da qualidade do vinho

Este projeto demonstra um pipeline de aprendizado de máquina para prever a qualidade do vinho com base em vários recursos químicos. O pipeline inclui o pré-processamento de dados, a seleção de modelos (Random Forest, K-Nearest Neighbors, SVM), o ajuste de hiperparâmetros e a implantação como uma API usando o Flask.

## Estrutura do projeto

### 1. API (servidor Flask)
O `app.py` contém o servidor Flask de backend responsável por servir um modelo de aprendizado de máquina que prevê a qualidade do vinho com base em recursos químicos. O modelo e o escalonador são carregados a partir de arquivos `.pkl` e usados para fazer previsões.

#### Arquivos principais:
- `app.py`: Define a API do Flask com a rota `/predict` para prever a qualidade do vinho com base em recursos.
- `best_model.pkl`: Contém o melhor modelo treinado.
- `scaler.pkl`: Contém o scaler usado para pré-processar os recursos.

#### Como executar:
```bash
# Instalar dependências
pip install -r requirements.txt
```

#### Executar o servidor Flask

```
python app.py
```

#### Endpoint da API:

```
POST /predict:
```


##### Request Body:
```
{
  “features": [7.88, 1.44, 0.81, 9.81, 0.47, 29, 44, 0.99208, 2.83, 1.31, 8.55]
}
```

##### Response:

```
{
  “prediction” (previsão): 5
}
```

### 2. Treinamento do modelo
O script train_model.py contém o pipeline de aprendizado de máquina para treinar o modelo no conjunto de dados Wine Quality, ajustar os hiperparâmetros e salvar o melhor modelo e escalonador para uso na produção.


##### Etapas principais:
Pré-processamento de dados: Dividir os dados em conjuntos de treinamento e teste.
Pipelines de modelos: Definição dos pipelines Random Forest, KNN e SVM com normalização e padronização.
GridSearchCV: execução de pesquisa de grade em hiperparâmetros para encontrar o melhor modelo.
Salvamento de modelo: Salvar o melhor modelo e escalonador usando joblib.
Como executar:

##### Instale as dependências

```
pip install -r requirements.txt
```

##### Treinar o modelo

```
python train_model.py
```

### 3. Interface (HTML)
O index.html fornece uma interface de usuário simples para inserir características do vinho e fazer previsões usando a API do Flask. O formulário permite que os usuários insiram valores manualmente ou gerem valores aleatórios com base nas propriedades químicas típicas do vinho.

#### Features:
Campos de entrada para todas as 11 características químicas.
Um botão para gerar valores válidos aleatórios para cada característica.
Exibição da qualidade prevista do vinho com base no resultado do modelo.

#### Como usar:

Abra o index.html em um navegador da Web.
Preencha o formulário ou gere valores aleatórios.
Envie o formulário para prever a qualidade do vinho.

### 4. Testes de unidade (Pytest)

Os testes de unidade são definidos para garantir que o modelo e as etapas de pré-processamento funcionem corretamente.

#### Testes incluídos:

Carregamento do modelo: Verifica se o modelo e o dimensionador estão carregados corretamente.
Desempenho do modelo: Garante que a precisão do modelo esteja acima de um determinado limite em um conjunto de dados de teste.
Comportamento do scaler: Garante que o dimensionador seja aplicado corretamente durante o pré-processamento.

#### Como executar:

```
# Instalar o pytest
pip install pytest
```

#### Execute os testes

pytest
Instalação
Clone o repositório atual:

Instale as bibliotecas Python necessárias:

```
pip install -r requirements.txt
```

Faça o download do conjunto de dados ou use o existente da UCI:

```
python train_model.py
```

## Tecnologias usadas
Flask: Para servir o modelo de aprendizado de máquina por meio de uma API.
Scikit-learn: Para treinar e testar modelos de aprendizado de máquina.
Joblib: Para salvar e carregar modelos.
Pytest: Para testes de unidade.
HTML/CSS: Para criar um frontend simples.

## Licença
Este projeto está licenciado sob a licença MIT.

