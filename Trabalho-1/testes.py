import json
import csv

def converter_json_para_csv(caminho_json, caminho_csv):
    try:
        # Tentar carregar o JSON e ignorar erros de decodificação
        with open(caminho_json, 'r', encoding='utf-8') as f:
            conteudo = f.read()
        
        # Tentar corrigir problemas de JSON malformado
        conteudo = conteudo.replace("][", ",")  # Exemplo de correção básica
        dados = json.loads(conteudo)  # Carregar JSON corrigido

        # Escrever dados no arquivo CSV
        with open(caminho_csv, 'w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            header = dados[0].keys()  # Cabeçalhos a partir da primeira linha
            csv_writer.writerow(header)  # Escrever cabeçalhos
            for item in dados:
                csv_writer.writerow(item.values())  # Escrever cada linha do JSON

        print(f"Conversão concluída! Arquivo CSV salvo em {caminho_csv}")

    except json.JSONDecodeError as e:
        print("Erro ao decodificar JSON:", e)
    except Exception as e:
        print("Erro:", e)

# Caminho dos arquivos
caminho_arquivo_json = 'Mineracao-de-dados/Trabalho-1/padaria_trab.json'
caminho_arquivo_csv = 'padaria_trab.csv'

# Executar a conversão
converter_json_para_csv(caminho_arquivo_json, caminho_arquivo_csv)
