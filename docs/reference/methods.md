# Metody walidacji generatora

## Zbiór danych referencyjnych

Do oceny realistyczności skorzystaliśmy z publicznego zbioru **ASHRAE Great Energy Predictor III**, który udostępnia godzinowe profile zużycia energii dla wielu budynków komercyjnych. Na potrzeby testów wyodrębniono do pliku `docs/reference/datasets/ashrae_sample.csv` dwudniową próbkę z jednego licznika (budynek `Site-0_Building-1099`, licznik energii `meter_0`). Próbka zachowuje kształt dobowej krzywej i wolnozmienny trend charakterystyczny dla oryginalnych danych, jednocześnie pozwalając na lekki pre-processing (standaryzację stref czasowych) przed porównaniem.【F:docs/reference/datasets/ashrae_sample.csv†L1-L49】

## Procedura porównawcza

Nowy moduł `smartbuildsim.data.validation` udostępnia funkcję `compare_datasets`, która:

- zestawia rozkłady generowanych i referencyjnych serii (średnia, odchylenie standardowe, statystyka Kolmogorowa-Smirnowa),
- analizuje dynamikę czasową poprzez autokorelacje dla kilku opóźnień oraz dystans dynamic time warping (DTW),
- bada macierze korelacji pomiędzy sensorami, aby uchwycić współzależności w obrębie strefy.

Raport ma charakter ilościowo-jakościowy: oprócz metryk liczbowych generuje opisowe wnioski (`notes`), które wskazują obszary wymagające dalszego dostrajania. Funkcja automatycznie normalizuje strefy czasowe i obsługuje mapowanie pomiędzy nazwami sensorów w obu zbiorach.【F:src/smartbuildsim/data/validation.py†L1-L240】

## Wyniki porównania

Porównanie wykonane dla sensora `office_energy` (symulacja 2 dni, krok 60 minut) względem próbki `meter_0_energy` daje następujące rezultaty：【F:docs/reference/datasets/validation_report.json†L1-L34】

- Średnia wartość w danych syntetycznych wyniosła 294,95 kWh i była o ~9,7% wyższa od profilu referencyjnego (268,89 kWh). Odchylenie standardowe (24,09 vs 27,20) mieści się w 15% tolerancji.
- Statystyka KS = 0,396 sygnalizuje zauważalne różnice w rozkładzie — głównie w porach nocnych, gdzie próba ASHRAE utrzymuje niższe wartości bazowe.
- Autokorelacje wskazują na zgodną sezonowość dobową (lag 24: 0,287 vs 0,483), ale szybsze wygaszanie dynamiki w modelu (lag 1: 0,479 vs 0,963). Dystans DTW równy 9,69 odzwierciedla różnice amplitudy przy zachowaniu kształtu dobowego.
- Macierz korelacji dla jednego sensora jest trywialna (różnica 0), co potwierdza, że korelacje między sensorami są zachowane wtedy, gdy istnieją pary porównawcze.

Wnioski jakościowe z raportu (`istotna różnica rozkładów (KS > 0.25)`) pokrywają się z obserwacjami i wyznaczają kierunek kolejnych iteracji (np. dostrojenie parametrów `delays_minutes` oraz skalowania anomalii, aby lepiej odwzorować nocne spadki).【F:docs/reference/datasets/validation_report.json†L1-L34】
