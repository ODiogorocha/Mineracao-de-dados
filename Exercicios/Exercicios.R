#Crie uma vetor de caracteres com a sequencia 1 ate 99. (vet1)
vet1 <- c()

for(i in 1:99){
  vet1[i] <- as.caractere(i)
}
vet1

#Crie uma matrix 4x4 com valores de 1 ate 16. (mat1)

mat1 <- matrix(1:16, nrow = 4, ncol = 4)

for(i in 1:4){
  for(j in 1:4){
    mat1[i , j] <- mat1[i , j]
  }
}
mat1

#Crie um data frame com a matriz anterior. (DF1)

DF1 <- as.data.frame(mat1)

#Coloque nomes nas colunas do data frame ('a','b','c','d') 

colnames(DF1) <- c("A", "B", "C")

#Crie uma lista com a,b,c. depois substitua o 'b' por 2. (list1)


