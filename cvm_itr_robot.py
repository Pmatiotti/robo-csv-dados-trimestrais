from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

WEBHOOK_URL = (
    "https://xlmvqhjwliamckyxlpfi.supabase.co/functions/v1/ingest-fundamental-data"
)

DRE_MAPPING = {
    "3.01": "revenue",
    "3.03": "gross_profit",
    "3.05": "ebit",
    "3.11": "net_income",
    "3.08": "depreciation_amortization",
}

BPA_MAPPING = {
    "1": "total_assets",
    "1.01.01": "cash_and_equivalents",
}

BPP_MAPPING = {
    "2.01.04": "emprestimos_cp",
    "2.02.01": "emprestimos_lp",
    "2.03": "total_equity",
}

FINANCIAL_SECTORS = [
    "Bancos",
    "Seguradoras",
    "Intermediários Financeiros",
    "Serviços Financeiros Diversos",
    "Holdings Diversificadas",
]


def extrair_trimestre(dt_refer: str) -> Tuple[int, int]:
    data = datetime.strptime(dt_refer, "%Y-%m-%d")
    ano = data.year
    mes = data.month

    if mes <= 3:
        trimestre = 1
    elif mes <= 6:
        trimestre = 2
    elif mes <= 9:
        trimestre = 3
    else:
        trimestre = 4

    return ano, trimestre


def is_financial_institution(setor_atividade: str) -> bool:
    if not setor_atividade:
        return False
    return any(sector.lower() in setor_atividade.lower() for sector in FINANCIAL_SECTORS)


def _scale_multiplier(scale: Optional[str]) -> float:
    if not scale:
        return 1.0
    scale = scale.strip().upper()
    if scale == "UNIDADE":
        return 1.0
    if scale == "MIL":
        return 1_000.0
    if scale == "MILHAO":
        return 1_000_000.0
    return 1.0


@dataclass
class ParsedStatement:
    data: pd.DataFrame
    columns: List[str]


class CVMITRRobot:
    def __init__(
        self, api_key: str, webhook_url: str = WEBHOOK_URL, filter_latest_exercise: bool = True
    ) -> None:
        self.api_key = api_key
        self.webhook_url = webhook_url
        self.cnpj_ticker_map: Dict[str, str] = {}
        self.filter_latest_exercise = filter_latest_exercise

    def load_cnpj_ticker_map(self, filepath: str) -> None:
        """Carrega mapeamento CNPJ → Ticker.

        Espera colunas: CNPJ_CIA e TICKER (ou CNPJ/Ticker).
        """
        df = pd.read_csv(filepath)
        normalized = {
            "cnpj_cia": "CNPJ_CIA",
            "cnpj": "CNPJ_CIA",
            "ticker": "TICKER",
        }
        df.rename(columns={col: normalized.get(col.lower(), col) for col in df.columns}, inplace=True)
        for _, row in df.iterrows():
            cnpj = str(row.get("CNPJ_CIA", "")).strip()
            ticker = str(row.get("TICKER", "")).strip()
            if cnpj and ticker:
                self.cnpj_ticker_map[cnpj] = ticker

    def parse_cadastro(self, filepath: str) -> pd.DataFrame:
        df = pd.read_csv(filepath, sep=";", encoding="latin-1")
        columns = ["CNPJ_CIA", "SETOR_ATIV"]
        available = [col for col in columns if col in df.columns]
        return df[available].drop_duplicates()

    def _read_statement(self, filepath: str) -> pd.DataFrame:
        return pd.read_csv(filepath, sep=";", encoding="latin-1")

    def _filter_latest(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.filter_latest_exercise and "ORDEM_EXERC" in df.columns:
            df = df[df["ORDEM_EXERC"].str.upper() == "ÚLTIMO"]
        return df

    def _pivot_statement(self, df: pd.DataFrame, mapping: Dict[str, str]) -> ParsedStatement:
        df = self._filter_latest(df)
        df = df[df["CD_CONTA"].isin(mapping.keys())].copy()
        if df.empty:
            return ParsedStatement(pd.DataFrame(columns=["CNPJ_CIA", "DT_REFER"]), [])

        df["VL_CONTA"] = df["VL_CONTA"].astype(float) * df["ESCALA_MOEDA"].apply(
            _scale_multiplier
        )
        df["mapped"] = df["CD_CONTA"].map(mapping)
        pivot = df.pivot_table(
            index=["CNPJ_CIA", "DT_REFER"],
            columns="mapped",
            values="VL_CONTA",
            aggfunc="sum",
        ).reset_index()
        columns = [col for col in pivot.columns if col not in {"CNPJ_CIA", "DT_REFER"}]
        return ParsedStatement(pivot, columns)

    def parse_dre(self, filepath: str) -> pd.DataFrame:
        df = self._read_statement(filepath)
        parsed = self._pivot_statement(df, DRE_MAPPING)
        if parsed.data.empty:
            return parsed.data
        parsed.data["ebitda"] = (
            parsed.data.get("ebit", 0).fillna(0)
            + parsed.data.get("depreciation_amortization", 0).fillna(0)
        )
        return parsed.data

    def parse_balanco_ativo(self, filepath: str) -> pd.DataFrame:
        df = self._read_statement(filepath)
        parsed = self._pivot_statement(df, BPA_MAPPING)
        return parsed.data

    def parse_balanco_passivo(self, filepath: str) -> pd.DataFrame:
        df = self._read_statement(filepath)
        parsed = self._pivot_statement(df, BPP_MAPPING)
        return parsed.data

    def merge_data(
        self, dre: pd.DataFrame, bpa: pd.DataFrame, bpp: pd.DataFrame, cadastro: pd.DataFrame
    ) -> pd.DataFrame:
        merged = dre.merge(bpa, on=["CNPJ_CIA", "DT_REFER"], how="outer")
        merged = merged.merge(bpp, on=["CNPJ_CIA", "DT_REFER"], how="outer")
        if not cadastro.empty:
            merged = merged.merge(cadastro, on="CNPJ_CIA", how="left")
        merged["total_debt"] = merged.get("emprestimos_cp", 0).fillna(0) + merged.get(
            "emprestimos_lp", 0
        ).fillna(0)
        merged["net_debt"] = merged.get("total_debt", 0).fillna(0) - merged.get(
            "cash_and_equivalents", 0
        ).fillna(0)
        return merged

    def calculate_indicators(self, row: pd.Series) -> pd.Series:
        receita = row.get("revenue", 0)
        lucro_bruto = row.get("gross_profit", 0)
        ebit = row.get("ebit", 0)
        ebitda = row.get("ebitda", 0)
        lucro_liquido = row.get("net_income", 0)

        patrimonio_liquido = row.get("total_equity", 0)
        ativo_total = row.get("total_assets", 0)

        if receita:
            row["gross_margin"] = lucro_bruto / receita
            row["ebit_margin"] = ebit / receita
            row["ebitda_margin"] = ebitda / receita if ebitda else None
            row["net_margin"] = lucro_liquido / receita

        if patrimonio_liquido:
            row["roe"] = lucro_liquido / patrimonio_liquido

        if ativo_total:
            row["roa"] = lucro_liquido / ativo_total

        return row

    def build_payload(self, df: pd.DataFrame) -> List[Dict]:
        companies: List[Dict] = []

        if df.empty:
            return companies

        df = df.replace([float("inf"), float("-inf")], pd.NA)

        def clean_value(value: object) -> Optional[float]:
            if pd.isna(value):
                return None
            if isinstance(value, (float, int)):
                return float(value)
            return value

        for cnpj, group in df.groupby("CNPJ_CIA"):
            ticker = self.cnpj_ticker_map.get(str(cnpj).strip())
            if not ticker:
                continue

            group_sorted = group.sort_values("DT_REFER")
            quarterly_history = []
            for _, row in group_sorted.iterrows():
                year, quarter = extrair_trimestre(row["DT_REFER"])

                quarterly_history.append(
                    {
                        "year": year,
                        "quarter": quarter,
                        "revenue": clean_value(row.get("revenue")),
                        "gross_profit": clean_value(row.get("gross_profit")),
                        "ebit": clean_value(row.get("ebit")),
                        "ebitda": clean_value(row.get("ebitda"))
                        if not is_financial_institution(row.get("SETOR_ATIV", ""))
                        else None,
                        "net_income": clean_value(row.get("net_income")),
                        "total_assets": clean_value(row.get("total_assets")),
                        "total_equity": clean_value(row.get("total_equity")),
                        "total_debt": clean_value(row.get("total_debt")),
                        "net_debt": clean_value(row.get("net_debt")),
                        "cash_and_equivalents": clean_value(row.get("cash_and_equivalents")),
                        "gross_margin": clean_value(row.get("gross_margin")),
                        "ebit_margin": clean_value(row.get("ebit_margin")),
                        "ebitda_margin": clean_value(row.get("ebitda_margin")),
                        "net_margin": clean_value(row.get("net_margin")),
                        "roe": clean_value(row.get("roe")),
                        "roa": clean_value(row.get("roa")),
                    }
                )

            companies.append(
                {
                    "ticker": ticker,
                    "asset_class": "acoes",
                    "is_financial": is_financial_institution(group_sorted["SETOR_ATIV"].iloc[0])
                    if "SETOR_ATIV" in group_sorted.columns
                    else False,
                    "data_source": "cvm_itr_bot",
                    "quarterly_history": quarterly_history,
                }
            )

        return companies

    def send_to_webhook(self, data: List[Dict], batch_size: int = 50) -> None:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
        }

        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            payload = {"data": batch}

            response = requests.post(self.webhook_url, json=payload, headers=headers, timeout=60)

            if response.status_code == 200:
                result = response.json()
                print(
                    f"Batch {i//batch_size + 1}: "
                    f"{result.get('quarterly_processed', 0)} registros"
                )
            else:
                print(f"Erro no batch {i//batch_size + 1}: {response.text}")

    def run(self, data_dir: str) -> None:
        print("1. Carregando mapeamento CNPJ → Ticker...")
        self.load_cnpj_ticker_map(f"{data_dir}/ticker_map.csv")

        print("2. Carregando cadastro de empresas...")
        cadastro = self.parse_cadastro(f"{data_dir}/itr_cia_aberta_2025.csv")

        print("3. Processando DRE...")
        dre = self.parse_dre(f"{data_dir}/itr_cia_aberta_DRE_con_2025.csv")

        print("4. Processando Balanço Ativo...")
        bpa = self.parse_balanco_ativo(f"{data_dir}/itr_cia_aberta_BPA_con_2025.csv")

        print("5. Processando Balanço Passivo...")
        bpp = self.parse_balanco_passivo(f"{data_dir}/itr_cia_aberta_BPP_con_2025.csv")

        print("6. Unificando dados...")
        merged = self.merge_data(dre, bpa, bpp, cadastro)

        print("7. Calculando indicadores...")
        merged = merged.apply(self.calculate_indicators, axis=1)

        print("8. Construindo payload...")
        payload = self.build_payload(merged)

        print(f"9. Enviando {len(payload)} empresas para o webhook...")
        self.send_to_webhook(payload)

        print("✅ Concluído!")


if __name__ == "__main__":
    robot = CVMITRRobot(api_key="SUA_INGEST_API_KEY")
    robot.run("./data/itr_2025")
