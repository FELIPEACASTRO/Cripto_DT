"""Ensemble evolutivo inspirado em Sakana AI para otimizacao de pesos via algoritmo genetico.

Usa merging evolutivo de modelos: um cromossomo codifica pesos (float 0-1)
e mascara binaria (incluir/excluir) para cada modelo candidato. A funcao
de fitness combina acuracia direcional (primaria) e Sharpe ratio (secundaria)
avaliadas no conjunto de validacao.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class EvolutionaryConfig:
    """Configuracao do algoritmo evolutivo."""

    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.1
    mutation_sigma: float = 0.1
    crossover_rate: float = 0.8
    elitism: int = 5
    tournament_k: int = 3


class EvolutionaryEnsemble(BaseModel):
    """Ensemble evolutivo que otimiza pesos e selecao de modelos via GA.

    Cada cromossomo contem:
        - weights: array de floats [0, 1] para cada modelo
        - mask: array binario indicando se o modelo esta ativo

    Fitness: directional_accuracy (primario) + 0.1 * sharpe_ratio (secundario)
    """

    def __init__(self, config: EvolutionaryConfig | None = None):
        self.config = config or EvolutionaryConfig()
        self.best_weights: np.ndarray | None = None
        self.best_mask: np.ndarray | None = None
        self.n_models: int = 0
        self.best_fitness: float = 0.0
        self.fitness_history: list[float] = []

    @property
    def name(self) -> str:
        return "evolutionary_ensemble"

    # ------------------------------------------------------------------
    # Representacao do cromossomo
    # ------------------------------------------------------------------

    def _init_population(self, n_models: int) -> list[dict]:
        """Inicializa populacao aleatoria de cromossomos."""
        population = []
        rng = np.random.default_rng()
        for _ in range(self.config.population_size):
            weights = rng.uniform(0.0, 1.0, size=n_models)
            # Mascara binaria: cada modelo tem ~70 % de chance de estar ativo
            mask = rng.random(n_models) < 0.7
            # Garantir pelo menos 1 modelo ativo
            if not mask.any():
                mask[rng.integers(n_models)] = True
            population.append({"weights": weights, "mask": mask})
        return population

    # ------------------------------------------------------------------
    # Fitness
    # ------------------------------------------------------------------

    def _fitness(
        self,
        chromosome: dict,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> float:
        """Calcula fitness: directional_accuracy + 0.1 * sharpe_ratio."""
        weights = chromosome["weights"]
        mask = chromosome["mask"]

        if not mask.any():
            return 0.0

        # Normalizar pesos ativos para somar 1
        active_weights = weights * mask.astype(float)
        total = active_weights.sum()
        if total == 0:
            return 0.0
        norm_weights = active_weights / total

        # Previsao ponderada
        preds = X_val @ norm_weights

        # Acuracia direcional (metrica primaria)
        dir_acc = float(np.mean(np.sign(preds) == np.sign(y_val)))

        # Sharpe ratio dos retornos previstos (metrica secundaria)
        if preds.std() > 1e-10:
            sharpe = float(preds.mean() / preds.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        return dir_acc + 0.1 * max(sharpe, 0.0)

    # ------------------------------------------------------------------
    # Operadores geneticos
    # ------------------------------------------------------------------

    def _tournament_selection(
        self,
        population: list[dict],
        fitnesses: np.ndarray,
        rng: np.random.Generator,
    ) -> dict:
        """Selecao por torneio com k=tournament_k."""
        indices = rng.choice(len(population), size=self.config.tournament_k, replace=False)
        best_idx = indices[np.argmax(fitnesses[indices])]
        return population[best_idx]

    def _crossover(
        self,
        parent_a: dict,
        parent_b: dict,
        rng: np.random.Generator,
    ) -> dict:
        """Crossover uniforme: para cada gene, escolhe de A ou B."""
        n = len(parent_a["weights"])
        cross_mask = rng.random(n) < 0.5

        child_weights = np.where(cross_mask, parent_a["weights"], parent_b["weights"])
        child_mask = np.where(cross_mask, parent_a["mask"], parent_b["mask"])

        # Garantir pelo menos 1 modelo ativo
        if not child_mask.any():
            child_mask[rng.integers(n)] = True

        return {"weights": child_weights, "mask": child_mask}

    def _mutate(self, chromosome: dict, rng: np.random.Generator) -> dict:
        """Mutacao Gaussiana nos pesos + flip de bits na mascara."""
        weights = chromosome["weights"].copy()
        mask = chromosome["mask"].copy()
        n = len(weights)

        for i in range(n):
            if rng.random() < self.config.mutation_rate:
                # Mutacao Gaussiana no peso
                weights[i] += rng.normal(0.0, self.config.mutation_sigma)
                weights[i] = np.clip(weights[i], 0.0, 1.0)

            if rng.random() < self.config.mutation_rate:
                # Flip da mascara
                mask[i] = not mask[i]

        # Garantir pelo menos 1 modelo ativo
        if not mask.any():
            mask[rng.integers(n)] = True

        return {"weights": weights, "mask": mask}

    # ------------------------------------------------------------------
    # Busca evolutiva
    # ------------------------------------------------------------------

    def _evolve(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Executa o algoritmo genetico e retorna (best_weights, best_mask)."""
        rng = np.random.default_rng()
        n_models = X_val.shape[1]
        population = self._init_population(n_models)

        best_overall_fitness = -np.inf
        best_overall_chromosome: dict | None = None

        for gen in range(self.config.generations):
            # Avaliar fitness de toda a populacao
            fitnesses = np.array([
                self._fitness(chrom, X_val, y_val) for chrom in population
            ])

            # Rastrear melhor
            gen_best_idx = int(np.argmax(fitnesses))
            gen_best_fitness = fitnesses[gen_best_idx]

            if gen_best_fitness > best_overall_fitness:
                best_overall_fitness = gen_best_fitness
                best_overall_chromosome = {
                    "weights": population[gen_best_idx]["weights"].copy(),
                    "mask": population[gen_best_idx]["mask"].copy(),
                }

            self.fitness_history.append(float(gen_best_fitness))

            if gen % 20 == 0:
                n_active = int(population[gen_best_idx]["mask"].sum())
                logger.info(
                    f"  Geracao {gen:3d}/{self.config.generations}: "
                    f"best_fitness={gen_best_fitness:.4f}, "
                    f"mean_fitness={fitnesses.mean():.4f}, "
                    f"active_models={n_active}/{n_models}"
                )

            # Elitismo: preservar os melhores
            elite_indices = np.argsort(fitnesses)[-self.config.elitism:]
            new_population = [
                {
                    "weights": population[i]["weights"].copy(),
                    "mask": population[i]["mask"].copy(),
                }
                for i in elite_indices
            ]

            # Gerar restante da populacao via selecao + crossover + mutacao
            while len(new_population) < self.config.population_size:
                parent_a = self._tournament_selection(population, fitnesses, rng)
                parent_b = self._tournament_selection(population, fitnesses, rng)

                if rng.random() < self.config.crossover_rate:
                    child = self._crossover(parent_a, parent_b, rng)
                else:
                    child = {
                        "weights": parent_a["weights"].copy(),
                        "mask": parent_a["mask"].copy(),
                    }

                child = self._mutate(child, rng)
                new_population.append(child)

            population = new_population

        assert best_overall_chromosome is not None
        logger.info(
            f"  Evolucao completa: best_fitness={best_overall_fitness:.4f}"
        )

        return best_overall_chromosome["weights"], best_overall_chromosome["mask"]

    # ------------------------------------------------------------------
    # Interface BaseModel
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Treina o ensemble evolutivo.

        X_train: shape (n_samples, n_models) -- previsoes dos modelos base
        y_train: targets reais
        X_val / y_val: dados de validacao (usados para fitness evolutivo)
        """
        self.n_models = X_train.shape[1]
        logger.info(
            f"  Iniciando busca evolutiva com {self.n_models} modelos, "
            f"pop={self.config.population_size}, gen={self.config.generations}"
        )

        # Se nao houver validacao, usar treino (menos ideal)
        eval_X = X_val if X_val is not None else X_train
        eval_y = y_val if y_val is not None else y_train

        self.best_weights, self.best_mask = self._evolve(eval_X, eval_y)

        # Metricas
        train_pred = self.predict(X_train)
        metrics: dict[str, float] = {
            "train_rmse": float(np.sqrt(np.mean((train_pred - y_train) ** 2))),
            "train_dir_acc": float(
                np.mean(np.sign(train_pred) == np.sign(y_train))
            ),
            "n_active_models": float(self.best_mask.sum()),
            "best_fitness": self.best_fitness,
        }

        if X_val is not None and y_val is not None:
            val_pred = self.predict(X_val)
            metrics["val_rmse"] = float(np.sqrt(np.mean((val_pred - y_val) ** 2)))
            metrics["val_dir_acc"] = float(
                np.mean(np.sign(val_pred) == np.sign(y_val))
            )

        # Log pesos ativos
        active_indices = np.where(self.best_mask)[0]
        active_w = self.best_weights[self.best_mask]
        if active_w.sum() > 0:
            norm_w = active_w / active_w.sum()
        else:
            norm_w = active_w
        for idx, w in zip(active_indices, norm_w):
            logger.info(f"    model_{idx}: peso={w:.4f}")

        logger.info(f"  {self.name} metricas: {metrics}")
        return metrics

    def _get_normalized_weights(self) -> np.ndarray:
        """Retorna pesos normalizados (somam 1) com mascara aplicada."""
        if self.best_weights is None or self.best_mask is None:
            raise RuntimeError("Modelo nao treinado. Chame fit() primeiro.")

        active_weights = self.best_weights * self.best_mask.astype(float)
        total = active_weights.sum()
        if total == 0:
            return np.ones(self.n_models) / self.n_models
        return active_weights / total

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Gera previsoes ponderadas.

        X: shape (n_samples, n_models) -- previsoes dos modelos base
        """
        norm_weights = self._get_normalized_weights()
        return X @ norm_weights

    def predict_with_confidence(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Previsoes com confianca baseada em concordancia e entropia dos pesos.

        Confianca combina:
        - Concordancia direcional entre modelos ativos
        - Entropia dos pesos (pesos concentrados = maior confianca)
        """
        norm_weights = self._get_normalized_weights()
        preds = X @ norm_weights

        # --- Concordancia direcional entre modelos ativos ---
        active_indices = np.where(self.best_mask)[0]
        if len(active_indices) > 1:
            active_preds = X[:, active_indices]
            # Fracao de modelos que concordam com a direcao do ensemble
            ensemble_sign = np.sign(preds).reshape(-1, 1)
            model_signs = np.sign(active_preds)
            agreement = np.mean(model_signs == ensemble_sign, axis=1)
        else:
            agreement = np.ones(len(X))

        # --- Entropia dos pesos (normalizada) ---
        active_w = norm_weights[self.best_mask]
        # Filtrar zeros para evitar log(0)
        active_w_pos = active_w[active_w > 1e-10]
        if len(active_w_pos) > 1:
            entropy = -np.sum(active_w_pos * np.log(active_w_pos))
            max_entropy = np.log(len(active_w_pos))
            # Baixa entropia = pesos concentrados = maior confianca
            weight_confidence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0
        else:
            weight_confidence = 1.0

        # Combinar: 70% concordancia, 30% concentracao de pesos
        confidence = 0.7 * agreement + 0.3 * weight_confidence
        confidence = np.clip(confidence, 0.0, 1.0)

        return preds, confidence

    def save(self, path: Path) -> None:
        """Salva o ensemble evolutivo em disco."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "config": self.config,
                "best_weights": self.best_weights,
                "best_mask": self.best_mask,
                "n_models": self.n_models,
                "best_fitness": self.best_fitness,
                "fitness_history": self.fitness_history,
            },
            path,
        )
        logger.info(f"Ensemble evolutivo salvo: {path}")

    @classmethod
    def load(cls, path: Path) -> "EvolutionaryEnsemble":
        """Carrega o ensemble evolutivo do disco."""
        data = joblib.load(path)
        instance = cls(config=data["config"])
        instance.best_weights = data["best_weights"]
        instance.best_mask = data["best_mask"]
        instance.n_models = data["n_models"]
        instance.best_fitness = data.get("best_fitness", 0.0)
        instance.fitness_history = data.get("fitness_history", [])
        return instance
