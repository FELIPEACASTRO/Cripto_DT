"""Selecao de features com Boruta, L1 e Mutual Information.

Implementa multiplos metodos de selecao de features para identificar
as variaveis mais relevantes para previsao de criptomoedas.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Lasso

logger = logging.getLogger(__name__)

# Verificar disponibilidade do Boruta
try:
    from boruta import BorutaPy

    _BORUTA_DISPONIVEL = True
except ImportError:
    _BORUTA_DISPONIVEL = False
    logger.warning(
        "BorutaPy nao encontrado. Instale com: pip install boruta. "
        "Metodo boruta_select usara fallback para mutual_info."
    )


class FeatureSelector:
    """Selecao de features com multiplos metodos.

    Metodos disponiveis:
    - Boruta: selecao baseada em Random Forest com features sombra
    - L1 (Lasso): selecao baseada em regularizacao L1
    - Mutual Information: selecao baseada em informacao mutua
    """

    def boruta_select(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        max_iter: int = 100,
    ) -> list[str]:
        """Seleciona features usando Boruta com Random Forest.

        Boruta cria copias aleatorias (sombras) de cada feature e compara
        a importancia das features originais com as sombras. Features que
        consistentemente superam suas sombras sao consideradas relevantes.

        Args:
            X: Matriz de features (n_samples, n_features)
            y: Vetor alvo
            feature_names: Nomes das features
            max_iter: Numero maximo de iteracoes do Boruta

        Returns:
            Lista de nomes das features confirmadas como relevantes
        """
        if not _BORUTA_DISPONIVEL:
            logger.warning(
                "Boruta nao disponivel, usando fallback para mutual_info_select"
            )
            return self.mutual_info_select(X, y, feature_names)

        logger.info(f"Executando Boruta com max_iter={max_iter}...")

        estimador = RandomForestRegressor(
            n_estimators=200,
            max_depth=7,
            n_jobs=-1,
            random_state=42,
        )

        try:
            boruta = BorutaPy(
                estimator=estimador,
                n_estimators="auto",
                max_iter=max_iter,
                random_state=42,
                verbose=0,
            )
            boruta.fit(X, y)

            # Features confirmadas (ranking == 1)
            selecionadas = [
                nome
                for nome, confirmada in zip(feature_names, boruta.support_)
                if confirmada
            ]

            # Features tentativas (borderline)
            tentativas = [
                nome
                for nome, tentativa in zip(feature_names, boruta.support_weak_)
                if tentativa
            ]

            logger.info(
                f"Boruta: {len(selecionadas)} features confirmadas, "
                f"{len(tentativas)} tentativas de {len(feature_names)} totais"
            )

            if tentativas:
                logger.info(f"  Features tentativas (borderline): {tentativas}")

            # Incluir tentativas junto com confirmadas
            resultado = selecionadas + tentativas

            if not resultado:
                logger.warning(
                    "Boruta nao confirmou nenhuma feature, "
                    "usando fallback para mutual_info_select"
                )
                return self.mutual_info_select(X, y, feature_names)

            return resultado

        except Exception as e:
            logger.error(f"Erro no Boruta: {e}. Usando fallback para mutual_info_select")
            return self.mutual_info_select(X, y, feature_names)

    def l1_select(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        alpha: float = 0.01,
    ) -> list[str]:
        """Seleciona features usando regularizacao L1 (Lasso).

        Lasso forca coeficientes irrelevantes a zero, efetivamente
        realizando selecao de features.

        Args:
            X: Matriz de features (n_samples, n_features)
            y: Vetor alvo
            feature_names: Nomes das features
            alpha: Forca da regularizacao (maior = mais features eliminadas)

        Returns:
            Lista de nomes das features com coeficiente nao-zero
        """
        logger.info(f"Executando selecao L1 (Lasso) com alpha={alpha}...")

        lasso = Lasso(alpha=alpha, max_iter=10000, random_state=42)
        lasso.fit(X, y)

        # Features com coeficiente nao-zero
        selecionadas = [
            nome
            for nome, coef in zip(feature_names, lasso.coef_)
            if abs(coef) > 1e-10
        ]

        logger.info(
            f"L1: {len(selecionadas)} features selecionadas "
            f"de {len(feature_names)} totais"
        )
        return selecionadas

    def mutual_info_select(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        k: int = 30,
    ) -> list[str]:
        """Seleciona top-k features por informacao mutua.

        Mutual information mede a dependencia entre cada feature e o alvo,
        capturando relacoes nao-lineares que correlacao linear nao detecta.

        Args:
            X: Matriz de features (n_samples, n_features)
            y: Vetor alvo
            feature_names: Nomes das features
            k: Numero de features a selecionar

        Returns:
            Lista de nomes das top-k features por informacao mutua
        """
        logger.info(f"Executando selecao por Mutual Information (top-{k})...")

        mi_scores = mutual_info_regression(X, y, random_state=42)

        # Ordenar por score decrescente e pegar top-k
        k = min(k, len(feature_names))
        indices_ordenados = np.argsort(mi_scores)[::-1][:k]

        selecionadas = [feature_names[i] for i in indices_ordenados]

        logger.info(
            f"Mutual Info: {len(selecionadas)} features selecionadas "
            f"de {len(feature_names)} totais"
        )
        return selecionadas

    def select_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        method: str = "boruta",
    ) -> list[str]:
        """Dispatcher que seleciona features pelo metodo especificado.

        Args:
            X: Matriz de features (n_samples, n_features)
            y: Vetor alvo
            feature_names: Nomes das features
            method: Metodo de selecao ('boruta', 'l1', 'mutual_info')

        Returns:
            Lista de nomes das features selecionadas
        """
        metodos = {
            "boruta": self.boruta_select,
            "l1": self.l1_select,
            "mutual_info": self.mutual_info_select,
        }

        if method not in metodos:
            logger.error(
                f"Metodo '{method}' nao reconhecido. "
                f"Opcoes: {list(metodos.keys())}. Usando mutual_info."
            )
            method = "mutual_info"

        # Se boruta solicitado mas nao disponivel, fallback
        if method == "boruta" and not _BORUTA_DISPONIVEL:
            logger.warning(
                "Boruta nao disponivel, usando fallback para mutual_info"
            )
            method = "mutual_info"

        logger.info(f"Selecionando features com metodo: {method}")
        return metodos[method](X, y, feature_names)

    def get_feature_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
    ) -> pd.DataFrame:
        """Calcula importancia das features por todos os metodos disponiveis.

        Retorna um DataFrame consolidado com a importancia e ranking
        de cada feature segundo cada metodo.

        Args:
            X: Matriz de features (n_samples, n_features)
            y: Vetor alvo
            feature_names: Nomes das features

        Returns:
            DataFrame com colunas: feature, rf_importance, mi_importance,
            l1_coef, rf_rank, mi_rank, l1_rank
        """
        logger.info("Calculando importancia das features por multiplos metodos...")

        resultado = pd.DataFrame({"feature": feature_names})

        # 1. Random Forest importance
        logger.info("  Calculando importancia via Random Forest...")
        rf = RandomForestRegressor(
            n_estimators=200, max_depth=7, n_jobs=-1, random_state=42
        )
        rf.fit(X, y)
        resultado["rf_importance"] = rf.feature_importances_
        resultado["rf_rank"] = resultado["rf_importance"].rank(ascending=False).astype(int)

        # 2. Mutual Information
        logger.info("  Calculando importancia via Mutual Information...")
        mi_scores = mutual_info_regression(X, y, random_state=42)
        resultado["mi_importance"] = mi_scores
        resultado["mi_rank"] = resultado["mi_importance"].rank(ascending=False).astype(int)

        # 3. L1 (Lasso) coeficientes
        logger.info("  Calculando importancia via L1 (Lasso)...")
        lasso = Lasso(alpha=0.01, max_iter=10000, random_state=42)
        lasso.fit(X, y)
        resultado["l1_coef"] = np.abs(lasso.coef_)
        resultado["l1_rank"] = resultado["l1_coef"].rank(ascending=False).astype(int)

        # 4. Boruta (se disponivel)
        if _BORUTA_DISPONIVEL:
            logger.info("  Calculando selecao via Boruta...")
            try:
                estimador = RandomForestRegressor(
                    n_estimators=200, max_depth=7, n_jobs=-1, random_state=42
                )
                boruta = BorutaPy(
                    estimator=estimador,
                    n_estimators="auto",
                    max_iter=50,  # Reduzido para analise rapida
                    random_state=42,
                    verbose=0,
                )
                boruta.fit(X, y)
                resultado["boruta_confirmada"] = boruta.support_
                resultado["boruta_tentativa"] = boruta.support_weak_
                resultado["boruta_rank"] = boruta.ranking_
            except Exception as e:
                logger.warning(f"  Erro no Boruta durante analise: {e}")

        # Ordenar por importancia media (RF + MI)
        resultado["media_rank"] = (resultado["rf_rank"] + resultado["mi_rank"]) / 2
        resultado = resultado.sort_values("media_rank").reset_index(drop=True)

        logger.info(
            f"Analise de importancia concluida para {len(feature_names)} features"
        )
        return resultado
