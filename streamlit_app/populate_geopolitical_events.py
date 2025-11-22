"""
Popula a tabela eventos_geopoliticos com dados simulados para demonstração
Execute: python populate_geopolitical_events.py
"""
import os
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import random
from dotenv import load_dotenv

load_dotenv()

# Configuração
DB_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/coinsight")
engine = create_engine(DB_URL)

# Eventos simulados realistas
EVENTOS_TEMPLATE = [
    # Eventos Econômicos
    {
        "pais_codigo": "US",
        "pais_nome": "Estados Unidos",
        "instituicao": "Federal Reserve",
        "titulo": "Fed aumenta taxa de juros em 0.25%",
        "descricao": "Federal Reserve eleva taxa básica de juros para controlar inflação",
        "categoria": "Econômico",
        "severidade": "Alto",
        "sentimento": "Negativo",
        "impacto_pct": -2.5,
        "confianca_pct": 85,
        "moedas": "BTC,ETH"
    },
    {
        "pais_codigo": "US",
        "pais_nome": "Estados Unidos",
        "instituicao": "SEC",
        "titulo": "SEC aprova ETF de Bitcoin à vista",
        "descricao": "Comissão de valores mobiliários aprova primeiro ETF spot de Bitcoin",
        "categoria": "Inovação",
        "severidade": "Alto",
        "sentimento": "Positivo",
        "impacto_pct": 8.5,
        "confianca_pct": 92,
        "moedas": "BTC"
    },
    {
        "pais_codigo": "CN",
        "pais_nome": "China",
        "instituicao": "Banco Popular da China",
        "titulo": "China anuncia yuan digital em expansão",
        "descricao": "PBoC expande programa piloto de moeda digital nacional",
        "categoria": "Inovação",
        "severidade": "Médio",
        "sentimento": "Neutro",
        "impacto_pct": 1.2,
        "confianca_pct": 78,
        "moedas": "BTC,ETH"
    },
    {
        "pais_codigo": "CN",
        "pais_nome": "China",
        "instituicao": "Governo Central",
        "titulo": "China reforça proibição de mineração de Bitcoin",
        "descricao": "Autoridades chinesas intensificam repressão à mineração de criptomoedas",
        "categoria": "Político",
        "severidade": "Alto",
        "sentimento": "Negativo",
        "impacto_pct": -5.8,
        "confianca_pct": 88,
        "moedas": "BTC"
    },
    # Eventos Políticos
    {
        "pais_codigo": "RU",
        "pais_nome": "Rússia",
        "instituicao": "Duma Estatal",
        "titulo": "Rússia legaliza criptomoedas para comércio exterior",
        "descricao": "Parlamento russo aprova uso de criptomoedas para transações internacionais",
        "categoria": "Político",
        "severidade": "Alto",
        "sentimento": "Positivo",
        "impacto_pct": 4.2,
        "confianca_pct": 82,
        "moedas": "BTC,ETH"
    },
    {
        "pais_codigo": "US",
        "pais_nome": "Estados Unidos",
        "instituicao": "Congresso",
        "titulo": "Senado discute regulamentação de stablecoins",
        "descricao": "Audiência sobre framework regulatório para moedas estáveis",
        "categoria": "Político",
        "severidade": "Médio",
        "sentimento": "Neutro",
        "impacto_pct": -0.8,
        "confianca_pct": 65,
        "moedas": "BTC,ETH,ADA"
    },
    # Eventos de Inovação
    {
        "pais_codigo": "US",
        "pais_nome": "Estados Unidos",
        "instituicao": "Ethereum Foundation",
        "titulo": "Ethereum completa upgrade para Proof of Stake",
        "descricao": "Rede Ethereum migra com sucesso para mecanismo de consenso mais eficiente",
        "categoria": "Inovação",
        "severidade": "Alto",
        "sentimento": "Positivo",
        "impacto_pct": 12.3,
        "confianca_pct": 95,
        "moedas": "ETH"
    },
    {
        "pais_codigo": "US",
        "pais_nome": "Estados Unidos",
        "instituicao": "Bitcoin Core",
        "titulo": "Bitcoin ativa Taproot upgrade",
        "descricao": "Atualização melhora privacidade e smart contracts no Bitcoin",
        "categoria": "Inovação",
        "severidade": "Médio",
        "sentimento": "Positivo",
        "impacto_pct": 3.5,
        "confianca_pct": 87,
        "moedas": "BTC"
    },
    # Mais eventos econômicos
    {
        "pais_codigo": "EU",
        "pais_nome": "União Europeia",
        "instituicao": "BCE",
        "titulo": "BCE mantém taxas de juros estáveis",
        "descricao": "Banco Central Europeu decide manter política monetária atual",
        "categoria": "Econômico",
        "severidade": "Baixo",
        "sentimento": "Neutro",
        "impacto_pct": 0.3,
        "confianca_pct": 70,
        "moedas": "BTC,ETH"
    },
    {
        "pais_codigo": "JP",
        "pais_nome": "Japão",
        "instituicao": "FSA",
        "titulo": "Japão aprova novas exchanges de criptomoedas",
        "descricao": "Agência reguladora japonesa licencia 5 novas corretoras",
        "categoria": "Econômico",
        "severidade": "Médio",
        "sentimento": "Positivo",
        "impacto_pct": 2.1,
        "confianca_pct": 76,
        "moedas": "BTC,ETH,ADA"
    },
    # Eventos geopolíticos
    {
        "pais_codigo": "UA",
        "pais_nome": "Ucrânia",
        "instituicao": "Governo",
        "titulo": "Ucrânia recebe doações em Bitcoin",
        "descricao": "País arrecada milhões em criptomoedas para apoio humanitário",
        "categoria": "Político",
        "severidade": "Alto",
        "sentimento": "Neutro",
        "impacto_pct": 1.8,
        "confianca_pct": 80,
        "moedas": "BTC,ETH"
    },
    {
        "pais_codigo": "SV",
        "pais_nome": "El Salvador",
        "instituicao": "Governo",
        "titulo": "El Salvador compra mais 500 Bitcoins",
        "descricao": "País reforça reservas nacionais de Bitcoin",
        "categoria": "Econômico",
        "severidade": "Médio",
        "sentimento": "Positivo",
        "impacto_pct": 1.5,
        "confianca_pct": 73,
        "moedas": "BTC"
    },
]


def criar_tabela():
    """Cria a tabela de eventos geopolíticos se não existir"""
    print("📊 Criando tabela eventos_geopoliticos...")
    
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS eventos_geopoliticos (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                pais_codigo TEXT,
                pais_nome TEXT,
                instituicao TEXT,
                titulo TEXT NOT NULL,
                descricao TEXT,
                categoria TEXT,
                severidade TEXT,
                sentimento TEXT,
                impacto_pct DOUBLE PRECISION,
                confianca_pct DOUBLE PRECISION,
                moedas TEXT
            );
            
            CREATE INDEX IF NOT EXISTS ix_eventos_timestamp 
            ON eventos_geopoliticos(timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS ix_eventos_categoria 
            ON eventos_geopoliticos(categoria);
        """))
    
    print("✅ Tabela criada/verificada")


def gerar_eventos_historicos(dias: int = 365):
    """Gera eventos históricos distribuídos ao longo do tempo"""
    print(f"🔄 Gerando eventos ao longo de {dias} dias...")
    
    eventos = []
    now = datetime.now()
    
    # Distribui eventos ao longo do período
    for i in range(dias):
        # Probabilidade de ter evento em cada dia (30% de chance)
        if random.random() < 0.3:
            # Escolhe evento aleatório do template
            evento = random.choice(EVENTOS_TEMPLATE).copy()
            
            # Define timestamp
            dias_atras = random.randint(0, dias)
            timestamp = now - timedelta(days=dias_atras, hours=random.randint(0, 23))
            evento['timestamp'] = timestamp
            
            # Adiciona variação no impacto
            if evento['impacto_pct']:
                variacao = random.uniform(0.8, 1.2)
                evento['impacto_pct'] = evento['impacto_pct'] * variacao
            
            eventos.append(evento)
    
    print(f"✅ {len(eventos)} eventos gerados")
    return eventos


def popular_banco(eventos):
    """Insere eventos no banco de dados"""
    print(f"💾 Inserindo {len(eventos)} eventos no banco...")
    
    df = pd.DataFrame(eventos)
    df.to_sql('eventos_geopoliticos', engine, if_exists='append', index=False)
    
    print("✅ Eventos inseridos com sucesso!")


def verificar_dados():
    """Verifica quantos eventos existem no banco"""
    with engine.begin() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM eventos_geopoliticos")).scalar()
        print(f"\n📊 Total de eventos no banco: {result}")
        
        # Estatísticas por categoria
        stats = pd.read_sql_query(text("""
            SELECT categoria, COUNT(*) as qtd, 
                   AVG(impacto_pct) as impacto_medio
            FROM eventos_geopoliticos
            GROUP BY categoria
            ORDER BY qtd DESC
        """), conn)
        
        print("\n📈 Eventos por categoria:")
        print(stats.to_string(index=False))


def limpar_tabela():
    """Remove todos os eventos (cuidado!)"""
    resposta = input("\n⚠️  Tem certeza que deseja LIMPAR todos os eventos? (sim/não): ")
    if resposta.lower() == 'sim':
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE TABLE eventos_geopoliticos RESTART IDENTITY"))
        print("✅ Tabela limpa!")
    else:
        print("❌ Operação cancelada")


def main():
    print("\n" + "="*60)
    print("🌍 POPULADOR DE EVENTOS GEOPOLÍTICOS")
    print("="*60 + "\n")
    
    print("Opções:")
    print("1. Popular com eventos históricos (365 dias)")
    print("2. Verificar dados existentes")
    print("3. Limpar todos os eventos")
    print("4. Sair")
    
    opcao = input("\nEscolha uma opção (1-4): ")
    
    if opcao == "1":
        criar_tabela()
        eventos = gerar_eventos_historicos(dias=365)
        popular_banco(eventos)
        verificar_dados()
        
        print("\n" + "="*60)
        print("✨ CONCLUÍDO COM SUCESSO!")
        print("="*60)
        print("\n💡 Dica: Agora você pode usar a aba 'Impacto Geopolítico'")
        print("   no Dashboard de IA para visualizar as correlações!\n")
    
    elif opcao == "2":
        verificar_dados()
    
    elif opcao == "3":
        limpar_tabela()
    
    else:
        print("👋 Até logo!")


if __name__ == "__main__":
    main()