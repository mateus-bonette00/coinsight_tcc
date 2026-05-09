import subprocess
import sys
import os

DB_NAME = "coinsight"
DB_USER = "postgres"
DB_PASS = "postgres"
DB_HOST = "localhost"
DB_PORT = "5432"

def passo(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")

def run(cmd, capture=False):
    result = subprocess.run(cmd, shell=True, capture_output=capture, text=True)
    return result

def criar_banco():
    passo("1. Criando banco de dados 'coinsight'...")
    r = run(f'sudo -u postgres psql -c "SELECT 1 FROM pg_database WHERE datname=\'{DB_NAME}\';"', capture=True)
    if "1 row" in r.stdout:
        print("✅ Banco já existe.")
        return
    r = run(f'sudo -u postgres psql -c "CREATE DATABASE {DB_NAME};"')
    if r.returncode == 0:
        print("✅ Banco criado com sucesso.")
    else:
        print("❌ Falha ao criar banco. Rode manualmente:")
        print(f"   sudo -u postgres psql -c \"CREATE DATABASE {DB_NAME};\"")
        print(f"   sudo -u postgres psql -c \"ALTER USER postgres WITH PASSWORD '{DB_PASS}';\"")

def definir_senha():
    passo("2. Definindo senha do usuário postgres...")
    r = run(f"sudo -u postgres psql -c \"ALTER USER postgres WITH PASSWORD '{DB_PASS}';\"")
    if r.returncode == 0:
        print("✅ Senha definida.")
    else:
        print("⚠️ Não foi possível definir a senha automaticamente.")

def rodar_etl():
    passo("3. Baixando dados históricos de BTC, ETH, ADA, SOL...")
    print("   Isso pode levar alguns minutos. Aguarde...")
    venv_python = os.path.join(
        os.path.dirname(__file__),
        "streamlit_app", ".venv", "bin", "python"
    )
    if not os.path.exists(venv_python):
        venv_python = sys.executable
    etl_path = os.path.join(os.path.dirname(__file__), "scripts", "etl_coins_ohlc.py")
    env = os.environ.copy()
    env["DATABASE_URL"] = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    r = subprocess.run([venv_python, etl_path], env=env)
    if r.returncode == 0:
        print("✅ Dados históricos carregados com sucesso.")
    else:
        print("❌ Falha no ETL. Verifique se o banco foi criado corretamente.")

def verificar():
    passo("4. Verificando dados no banco...")
    venv_python = os.path.join(
        os.path.dirname(__file__),
        "streamlit_app", ".venv", "bin", "python"
    )
    if not os.path.exists(venv_python):
        venv_python = sys.executable
    check = os.path.join(os.path.dirname(__file__), "check_db.py")
    if os.path.exists(check):
        subprocess.run([venv_python, check])

def instrucoes_finais():
    passo("✅ SETUP CONCLUÍDO!")
    print("""
Para rodar o projeto:

  cd ~/Documentos/Projetos/meus-projetos/coinsight_tcc/streamlit_app
  source .venv/bin/activate
  streamlit run app.py

O navegador vai abrir em: http://localhost:8501
""")

if __name__ == "__main__":
    print("\n🚀 Configurando o projeto CoinSight TCC...\n")
    criar_banco()
    definir_senha()
    rodar_etl()
    verificar()
    instrucoes_finais()
