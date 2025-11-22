#!/usr/bin/env python3
"""Script para verificar o estado do banco de dados"""
from sqlalchemy import create_engine, text, inspect
import pandas as pd

# Conexão
engine = create_engine('postgresql+psycopg2://postgres:postgres@localhost:5432/coinsight')

print("\n" + "="*60)
print("🔍 VERIFICAÇÃO DO BANCO DE DADOS - COINSIGHT")
print("="*60 + "\n")

# 1. Listar tabelas
print("📋 TABELAS EXISTENTES:")
print("-" * 60)
inspector = inspect(engine)
tabelas = inspector.get_table_names()
if tabelas:
    for i, tabela in enumerate(tabelas, 1):
        print(f"  {i}. {tabela}")
else:
    print("  ⚠️  Nenhuma tabela encontrada!")
print()

# 2. Verificar dados em cada tabela
print("📊 QUANTIDADE DE REGISTROS:")
print("-" * 60)
with engine.connect() as conn:
    for tabela in tabelas:
        try:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {tabela}")).scalar()
            print(f"  {tabela:.<40} {result:>10} registros")
        except Exception as e:
            print(f"  {tabela:.<40} {'ERRO':>10}")

print()

# 3. Verificar tabela de preços especificamente (mais importante)
print("💰 DETALHES DA TABELA 'precos' (se existir):")
print("-" * 60)
if 'precos' in tabelas:
    try:
        with engine.connect() as conn:
            # Contar por moeda
            df_moedas = pd.read_sql_query(text("""
                SELECT moeda_id, COUNT(*) as registros,
                       MIN(timestamp) as primeiro_registro,
                       MAX(timestamp) as ultimo_registro
                FROM precos
                GROUP BY moeda_id
                ORDER BY moeda_id
            """), conn)

            if not df_moedas.empty:
                print(df_moedas.to_string(index=False))
                print()

                # Verificar quais colunas existem
                colunas = inspector.get_columns('precos')
                print(f"  Colunas: {', '.join([c['name'] for c in colunas])}")
            else:
                print("  ⚠️  Tabela 'precos' existe mas está VAZIA!")
    except Exception as e:
        print(f"  ❌ Erro ao consultar: {e}")
else:
    print("  ⚠️  Tabela 'precos' NÃO EXISTE!")

print()

# 4. Verificar tabela de eventos geopolíticos
print("🌍 DETALHES DA TABELA 'eventos_geopoliticos' (se existir):")
print("-" * 60)
if 'eventos_geopoliticos' in tabelas:
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM eventos_geopoliticos")).scalar()
            if result > 0:
                df_eventos = pd.read_sql_query(text("""
                    SELECT categoria, COUNT(*) as qtd
                    FROM eventos_geopoliticos
                    GROUP BY categoria
                    ORDER BY qtd DESC
                """), conn)
                print(f"  Total de eventos: {result}")
                print(f"\n  Por categoria:")
                print(df_eventos.to_string(index=False))
            else:
                print("  ⚠️  Tabela existe mas está VAZIA!")
    except Exception as e:
        print(f"  ❌ Erro ao consultar: {e}")
else:
    print("  ⚠️  Tabela 'eventos_geopoliticos' NÃO EXISTE!")

print()

# 5. Verificar tabela de moedas
print("🪙 DETALHES DA TABELA 'moedas' (se existir):")
print("-" * 60)
if 'moedas' in tabelas:
    try:
        with engine.connect() as conn:
            df_moedas = pd.read_sql_query(text("""
                SELECT id, simbolo, nome, ativo
                FROM moedas
                ORDER BY id
            """), conn)
            if not df_moedas.empty:
                print(df_moedas.to_string(index=False))
            else:
                print("  ⚠️  Tabela existe mas está VAZIA!")
    except Exception as e:
        print(f"  ❌ Erro ao consultar: {e}")
else:
    print("  ⚠️  Tabela 'moedas' NÃO EXISTE!")

print("\n" + "="*60)
print("✅ VERIFICAÇÃO CONCLUÍDA")
print("="*60 + "\n")
