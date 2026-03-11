import argparse
import sys
import os
import unittest

# Aggiungiamo la root del progetto al path per permettere gli import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_process_mode():
    print("\n=== ESECUZIONE BENCHMARK COSTRUZIONE (Processi) ===")
    try:
        from tests import test
        test.main()
    except ImportError:
        try:
            import test
            test.main()
        except ImportError as e:
            print(f"Errore: Impossibile trovare il modulo di test costruzione. {e}")

def run_thread_mode():
    print("\n=== ESECUZIONE BENCHMARK COSTRUZIONE (Thread) ===")
    try:
        from tests import test_no_gil
        test_no_gil.main()
    except:
        try:
            import test_threading
            test_threading.main()
        except ImportError as e:
            print(f"Errore: Impossibile trovare il modulo di test costruzione. {e}")


def run_query_mode():
    print("\n=== ESECUZIONE BENCHMARK QUERY ===")
    try:
        from tests import test_queries
        test_queries.run_query_benchmark()
    except ImportError:
        try:
            import test_queries  # Vecchio nome (se nella root)
            test_queries.run_query_benchmark()
        except ImportError as e:
            print(f"Errore: Impossibile trovare il modulo di test query. {e}")

def run_unit_tests():
    print("\n=== ESECUZIONE UNIT TESTS (Correttezza e Consistenza) ===")

    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'unittesting')

    # Carichiamo tutti i test che iniziano con 'verify_'
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=test_dir, pattern='verify_*.py')

    # Eseguiamo i test
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Se i test falliscono, possiamo decidere di uscire con un errore
    if not result.wasSuccessful():
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Bloom Filter Project Runner")
    
    parser.add_argument(
        "mode", 
        choices=["process", "threading", "query", "unit", "all"],
        help="Modalità di esecuzione: 'construction' (processi), 'threading' (thread), 'query' (interrogazioni),'test' (unit tests), 'all' (tutto)"
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    if args.mode == "process":
        run_process_mode()
    elif args.mode == "threading":
        run_thread_mode()
    elif args.mode == "query":
        run_query_mode()
    elif args.mode == "unit":
        run_unit_tests()
    elif args.mode == "all":
        run_process_mode()
        run_thread_mode()
        run_query_mode()

if __name__ == "__main__":
    main()
