install.packages("arules")  

library(arules)

# Função para carregar dados do CSV
carregar_dados <- function(arquivo_csv) {
  dados <- read.csv(arquivo_csv, header = FALSE, stringsAsFactors = FALSE)
  
  # Separar os produtos em cada compra
  dados$V2 <- strsplit(dados$V2, ", ")
  
  # Converter os dados em um formato transacional
  transacoes <- as(dados$V2, "transactions")
  
  return(transacoes)
}

# Função para encontrar regras de associação
encontrar_regras_associacao <- function(transacoes, suporte_minimo = 0.05, confianca_minima = 0.5) {
  regras <- apriori(transacoes, parameter = list(supp = suporte_minimo, conf = confianca_minima))
  return(regras)
}

# Função principal
main <- function() {
  arquivo_csv <- "/home/diogo/Documents/Aulas/M.dados/Mineracao-de-dados/Trabalho-1/padaria.csv"  
  
  # Carregar dados
  transacoes <- carregar_dados(arquivo_csv)
  
  regras <- encontrar_regras_associacao(transacoes)
  
  # Exibir as 5 principais regras
  cat("As 5 principais regras de associação:\n")
  if (length(regras) > 0) {
    regras_principais <- head(sort(regras, by = "confidence"), 5)
    inspect(regras_principais)
  } else {
    cat("Nenhuma regra encontrada.\n")
  }
  
  # Regras que implicam a compra de "Doce"
  regras_doce <- subset(regras, rhs %pin% "Doce")
  cat("\nRegras que implicam a compra de 'Doce':\n")
  
  if (length(regras_doce) > 0) {
    inspect(regras_doce)
  } else {
    cat("Nenhuma regra que implica a compra de 'Doce' encontrada.\n")
  }
  
  # Verificar se há menos de 5 regras e tentar incluir a regra do "Doce"
  if (length(regras_principais) < 5 && length(regras_doce) > 0) {
    cat("\nAdicionando regras relacionadas ao 'Doce' para completar as 5 principais regras:\n")
    regras_principais <- unique(c(regras_principais, regras_doce))
    regras_principais <- head(sort(regras_principais, by = "confidence"), 5)
    inspect(regras_principais)
  }
}

main()
