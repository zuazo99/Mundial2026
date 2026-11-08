# Registro de errores

Errores cometidos al trabajar en este repo, con su causa y la regla para no repetirlos.
**Consultar antes de empezar una tarea y añadir una entrada cada vez que se cometa un error nuevo**
(incluidos los pequeños: lo que cuesta caro es repetirlos).

Formato de cada entrada: *qué pasó → causa → cómo se detectó → regla*.

## Reglas de oro (resumen)

1. **Comprobar que cada paso funcionó antes de usar su resultado** (código de salida, existencia del fichero), sobre todo antes de comparar contra una referencia.
2. **Validar los datos de prueba antes de citar métricas** obtenidas con ellos; un generador sintético también necesita sus propios tests de invariantes.
3. **Comprobar que un test puede fallar** (mutación rápida: romper el código y ver que el test lo detecta).
4. **Medir la cobertura de nombres/alias entre fuentes**; un emparejamiento fallido cae en silencio a un valor de respaldo.
5. **No dar nada por confirmado sin haberlo mirado** (CI, cifras, "funciona"); si no se ha comprobado, decirlo.
6. **Leer las instrucciones del usuario literalmente**; si hay dos lecturas posibles, preguntar o releer antes de actuar.
7. **Al reescribir documentación, conservar las notas que el usuario pidió expresamente.**
8. **Intérprete y entorno**: comprobar con `which`/`--version` qué Python y qué `pip` se usan.
9. **Operaciones destructivas** (reset duro, borrar ramas, force-push): no intentarlas sin confirmación explícita; hay alternativas no destructivas.
10. **Validar lo que se mide contra lo que se usa**: una comprobación de calibración con ventanas reales no valida un estado congelado.

Registros: [`2026-11_sesion_liga.md`](2026-11_sesion_liga.md).
