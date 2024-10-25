# Instalação e carregamento dos pacotes necessários
install.packages("arules")  
install.packages("readr")
install.packages("dplyr")
install.packages("hunspell")

library(arules)
library(readr)
library(dplyr)
library(hunspell)

# Carregando o arquivo CSV (substitua o caminho pelo correto)
caminho_arquivo <- "Mineracao-de-dados/Trabalho-1/padaria.csv" 
dados <- read_csv(caminho_arquivo)

# Visualizando as primeiras linhas dos dados
print(head(dados))

# Função para verificar erros de português em uma coluna específica
verificar_portugues <- function(texto) {
  # Retorna TRUE se houver erro
  !hunspell::hunspell_check(texto)
}

# Filtrando os dados para excluir linhas com erros de português na coluna 'produtos'
dados_filtrados <- dados %>%
  filter(!verificar_portugues(produtos))

# Visualizando os dados filtrados
print(head(dados_filtrados))

# Convertendo o dataframe filtrado para uma classe de transações
# Supondo que o CSV tenha colunas: compra e produtos
transacoes <- as(split(dados_filtrados$produtos, dados_filtrados$compra), "transactions")

# Verificando as transações
summary(transacoes)

# Extração das regras de associação
# Definindo os parâmetros de suporte e confiança
regras <- apriori(transacoes, parameter = list(support = 0.01, confidence = 0.5))

# Ordenando e selecionando as 5 principais regras com base no lift
regras_principais <- head(sort(regras, by = "lift"), 5)
cat("As 5 principais regras de associação são:\n")
inspect(regras_principais)

# Encontrando regras que impliquem na compra de "Doce"
regras_doce <- subset(regras, rhs %in% "Doce")
cat("\nRegras que implicam na compra de 'Doce':\n")
inspect(regras_doce)

# Encontrando o produto mais influente (1 para 1)
regras_1_para_1 <- subset(regras, lhs %pin% c("Café", "Pão", "Presunto", "Queijo", "Pastel", "Doce", "Refri") & size(rhs) == 1)
regras_1_para_1 <- head(sort(regras_1_para_1, by = "confidence"), 5)
cat("\nRegras 1 para 1 (prodA => prodB):\n")
inspect(regras_1_para_1)

# Exportando as regras para arquivos CSV para o relatório
write(regras_principais, file = "regras_principais.csv", sep = ",", quote = TRUE, row.names = FALSE)
write(regras_doce, file = "regras_doce.csv", sep = ",", quote = TRUE, row.names = FALSE)

# Finalizando
cat("\nProcesso completo! As regras foram exportadas para 'regras_principais.csv' e 'regras_doce.csv'.")
