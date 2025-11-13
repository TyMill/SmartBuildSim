# Model benchmarks

Nowoczesne modele i procedury porównawcze są dostępne poprzez moduł
`smartbuildsim.evaluation.benchmark`. Wykorzystujemy wspólny zestaw danych
syntetycznych (`office-small`) z wymuszoną częstością anomalii, aby ocenić
regresję zużycia energii, detekcję anomalii oraz sterowanie RL.

## Modele

- **Regresja**: bazowa regresja liniowa (pipeline `StandardScaler` +
  `LinearRegression`) oraz mocniejszy `HistGradientBoostingRegressor`.
- **Anomalie**: `IsolationForest` jako baseline oraz `LocalOutlierFactor`
  (`novelty=True`) jako gęstościowy model sąsiedzki.
- **RL**: tablicowy Q-learning (`smartbuildsim.models.rl.train_policy`) i nowy
  soft-Q (`train_soft_q_policy`) inspirowany SAC z regularyzacją entropii.

## Projekt eksperymentów

- Każdy moduł wykorzystuje 5-krotną walidację z tasowaniem powtarzaną dla kilku
  seedów (`[0, 1, 2]` dla regresji/anomalii, `[7, 11, 21, 42]` dla RL) z
  rozwiązywaniem przez [`smartbuildsim.config`](../determinism.md).
- Pipeline’y łączą różne strategie skalowania (`StandardScaler`,
  `MinMaxScaler`, brak skalowania) i porównują średnie RMSE/F1.
- Wyniki raportują średnią i odchylenie standardowe, a także testy istotności
  (t-test i Wilcoxona) między baseline’em a modelami alternatywnymi.
- Analiza wrażliwości obejmuje agregację wyników względem skalowania, aby
  wskazać stabilność modeli na zmianę jednostek.

Procedura jest zautomatyzowana w skrypcie
`examples/scripts/run_benchmarks.py`, który generuje raport JSON.

## Wyniki

Podsumowanie z `docs/reference/datasets/benchmark_report.json` przedstawiono w
Tabeli 1. Wyniki regresji pokazują, że gradient boosting nie pokonuje liniowej
regresji na krótkim horyzoncie — różnice są statystycznie nieistotne.
`IsolationForest` przewyższa `LOF` przy umiarkowanej liczbie anomalii, a soft-Q
poprawia stabilność (niższe odchylenie) przy podobnej średniej nagrodzie.

| Moduł | Najlepszy model | Średnia ± std | Wpływ normalizacji |
|-------|-----------------|---------------|---------------------|
| Regresja | LinearRegression + StandardScaler | 21.27 ± 1.57 RMSE | RMSE baseline spada do 0.82 (Δ −20.45) |
| Anomalie | IsolationForest + StandardScaler | 0.17 ± 0.18 F1 | Brak zmiany (Δ 0.00) |
| RL | Soft-Q (entropy) | 0.53 ± 0.01 reward | — |

Dodatkowe szczegóły (wartości dla wszystkich kombinacji modeli i skalowania
oraz metryki testów istotności) znajdują się bezpośrednio w raporcie JSON.

### Wpływ normalizacji

Wprowadzenie standaryzacji na poziomie generatora powoduje, że pipeline
regresyjny operuje na bezwymiarowych wartościach energii. Średnie RMSE dla
baseline’u `linear+standard` spada z 21.27 do 0.82, a wariancja maleje ponad
dwudziestokrotnie. Różnice między wariantami skalowania pozostają jednak
nieistotne statystycznie — standaryzacja głównie harmonizuje jednostki, bez
zmiany rankingu modeli.【F:docs/reference/datasets/benchmark_report.json†L3-L159】

W detekcji anomalii normalizacja nie zmienia średniego F1 izolacyjnego lasu –
wynik 0.17 utrzymuje się niezależnie od przeskalowania, co oznacza, że model
jest odporny na zmiany jednostek w syntetycznym zestawie danych. Metryki LOF
również pozostają praktycznie niezmienione, co sugeruje, że najważniejsze jest
zachowanie dynamiki czasowej, a nie skala wartości.【F:docs/reference/datasets/benchmark_report.json†L161-L318】
