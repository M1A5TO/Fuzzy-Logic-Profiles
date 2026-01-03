"""
Prosty przykład odbierania i wysyłania wiadomości do RabbitMQ.
"""

import os
import json
import logging
import signal
import sys
import subprocess
from dotenv import load_dotenv
import pika
import requests
# #########TEST##########
# from api_client import APIClient, AuthClient
# #########TEST##########

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment
RABBITMQ_HOST = os.getenv('RABBITMQ_HOST', 'localhost')
RABBITMQ_PORT = int(os.getenv('RABBITMQ_PORT', 5672))
RABBITMQ_USER = os.getenv('RABBITMQ_DEFAULT_USER', 'default')
RABBITMQ_PASSWORD = os.getenv('RABBITMQ_DEFAULT_PASS', 'default')
RABBITMQ_URL = os.getenv('RABBITMQ_URL', 'default')
API_BASE_URL = os.getenv("API_BASE_URL", "default")
# API_USERNAME = os.getenv("API_USERNAME", "guest")
# API_PASSWORD = os.getenv("API_PASSWORD", "guest")
INPUT_QUEUE = os.getenv('INPUT_QUEUE', 'scraper_new_offers')



class ProcessingTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise ProcessingTimeout()


# #########TEST##########
# def build_api_client() -> APIClient:
#     base_url = "http://localhost:8081"
#
#     auth = AuthClient(base_url)
#     username = API_USERNAME
#     password = API_PASSWORD
#
#     # jeśli API ma auth → logujemy
#     if username and password:
#         if not auth.login(username, password):
#             raise RuntimeError("API login failed (check API_USERNAME / API_PASSWORD)")
#
#     return APIClient(base_url, auth_client=auth)
# #########TEST##########


class RabbitMQProcessor:
    """Prosty procesor wiadomości RabbitMQ."""

    def __init__(self):
        self.connection = None
        self.channel = None
    # #########TEST##########
    #     self._api: APIClient | None = None
    #
    # @property
    # def api(self) -> APIClient:
    #     if self._api is None:
    #         self._api = build_api_client()
    #     return self._api
    # #########TEST##########

    def connect(self):
        """Połącz się z RabbitMQ."""
        try:
            credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASSWORD)
            parameters = pika.ConnectionParameters(
                host=RABBITMQ_HOST,
                port=RABBITMQ_PORT,
                credentials=credentials
            )

            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()
            self.channel.basic_qos(prefetch_count=1)

            # Utwórz kolejki (jeśli nie istnieją)
            self.channel.queue_declare(queue=INPUT_QUEUE, durable=True)

            logger.info(f"Połączono z RabbitMQ: {RABBITMQ_HOST}:{RABBITMQ_PORT}")
        except Exception as e:
            logger.error(f"Błąd połączenia z RabbitMQ: {e}")
            raise

    # =========================
    #  CHANGED: process_message
    # =========================
    def process_message(self, message_data: dict) -> dict:
        apartment_id = message_data.get("apartment_id")

        if apartment_id is None:
            logger.error("Missing apartment_id (reason=missing field). input=%s", message_data)
            raise Exception("Missing apartment_id")

        try:
            apartment_id_int = int(apartment_id)
        except Exception:
            logger.error("Invalid apartment_id (reason=not int). input=%s", message_data)
            raise Exception("Invalid apartment_id")

        # 1) Pobierz mieszkanie z API
        try:
            apartment = self.api_get(f"/apartments/{apartment_id_int}")
        except Exception as e:
            logger.error("API error for apartment_id=%s (reason=%s)", apartment_id_int, e)
            raise

        # 2) Pobierz relacje POI z API: GET /apartments/{id}/pois
        try:
            poi_rels = self.api_get(f"/apartments/{apartment_id_int}/pois")
        except Exception as e:
            logger.error("API error for apartment_id=%s POIs (reason=%s)", apartment_id_int, e)
            raise

        if not isinstance(poi_rels, list):
            raise Exception("GET /apartments/{id}/pois must return a list")

        # 3) Znormalizuj format relacji do tego, czego oczekuje score_apartments_offline.py
        # Expected per rel: { apartment_id, time_to_poi, poi:{category,...} or category }
        normalized = []
        for rel in poi_rels:
            if not isinstance(rel, dict):
                continue

            ttp = rel.get("time_to_poi")
            poi_obj = rel.get("poi") if isinstance(rel.get("poi"), dict) else None

            # minimalne wymagania dla scorer-a:
            # - apartment_id
            # - time_to_poi
            # - poi.category (albo rel.category)
            cat = None
            if poi_obj:
                cat = poi_obj.get("category")
            if cat is None:
                cat = rel.get("category")

            if ttp is None or cat is None:
                continue

            normalized.append({
                "apartment_id": apartment_id_int,
                "time_to_poi": ttp,
                "poi": {"category": cat},
            })

        if not normalized:
            # scorer zadziała, ale da POI_score ~ 0 (i spadną wyniki); to zwykle oznacza błąd pipeline
            raise Exception(f"No valid POI relations for apartment_id={apartment_id_int} from API")

        # 4) Przygotuj input dla score_apartments_offline.py (lista z 1 mieszkaniem) + tmp rels json
        work_dir = os.getenv("WORK_DIR", ".")
        apartments_tmp = os.path.join(work_dir, f"_tmp_apartment_{apartment_id_int}.json")
        rels_tmp = os.path.join(work_dir, f"_tmp_poi_rels_{apartment_id_int}.json")
        out_tmp = os.path.join(work_dir, f"_tmp_scored_{apartment_id_int}.json")

        with open(apartments_tmp, "w", encoding="utf-8") as f:
            json.dump([apartment], f, ensure_ascii=False, indent=2)

        with open(rels_tmp, "w", encoding="utf-8") as f:
            json.dump(normalized, f, ensure_ascii=False, indent=2)

        # 5) Uruchom Twoje offline scoring (bez zmian w score_apartments_offline.py)
        scorer_script = os.getenv("SCORE_SCRIPT", "score_apartments_offline.py")

        cmd = [
            sys.executable, scorer_script,
            "--apartments", apartments_tmp,
            "--poi-rels", rels_tmp,
            "--out", out_tmp,
        ]

        p = subprocess.run(cmd, text=True, capture_output=True)

        if p.returncode != 0:
            logger.error(
                "Scoring error for apartment_id=%s (reason=%s). stderr_tail=%s",
                apartment_id_int,
                f"exit code {p.returncode}",
                (p.stderr or "")[-800:]
            )
            raise Exception(f"Scoring error exit_code={p.returncode}")

        # 6) Wczytaj wynik i wyciągnij pola do aktualizacji
        try:
            with open(out_tmp, "r", encoding="utf-8") as f:
                scored_list = json.load(f)
        except Exception as e:
            logger.error("Scoring output read error for apartment_id=%s (reason=%s)", apartment_id_int, e)
            raise Exception("Scoring output invalid JSON") from e

        if not isinstance(scored_list, list) or not scored_list:
            raise Exception("Scoring output is not a non-empty list")

        scored = scored_list[0]
        if not isinstance(scored, dict):
            raise Exception("Scoring output first element is not a dict")

        fields_to_update = [
            "student_attractiveness",
            "single_attractiveness",
            "dog_owner_attractiveness",
            "family_attractiveness",
            "universal_attractiveness",
            "poi_desc",
            "price_desc",
            "size_desc",
        ]

        payload = {}
        for k in fields_to_update:
            if k in scored:
                payload[k] = scored[k]

        if not payload:
            raise Exception("No score fields found in scoring output")

        # 7) Zapisz do backendu przez PUT
        try:
            self.api_put(f"/apartments/{apartment_id_int}", payload)
        except Exception as e:
            logger.error("API PUT error for apartment_id=%s (reason=%s)", apartment_id_int, e)
            raise

        logger.info("Processed OK for apartment_id=%s", apartment_id_int)
        return {"apartment_id": apartment_id_int, "scores": payload}

    def api_post(self, path: str, payload: dict) -> dict:
        url = f"{API_BASE_URL}{path}"
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        return r.json() if r.content else {}

    def api_get(self, path: str, params: dict | None = None) -> dict:
        url = f"{API_BASE_URL}{path}"
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json() if r.content else {}

    # =====================
    #  ADDED: api_put
    # =====================
    def api_put(self, path: str, payload: dict) -> dict:
        url = f"{API_BASE_URL}{path}"
        r = requests.put(url, json=payload, timeout=30)
        r.raise_for_status()
        return r.json() if r.content else {}

    # ##########TEST##########
    # def api_get(self, path: str, params: dict | None = None) -> dict:
    #     resp = self.api.get(path)
    #     return resp if resp is not None else {}
    #
    # def api_post(self, path: str, payload: dict) -> dict:
    #     resp = self.api.post(path, payload)
    #     return resp if resp is not None else {}
    # ###########TEST##########

    # =========================
    #  CHANGED: on_message
    # =========================
    def on_message(self, ch, method, properties, body):
        """Obsługa otrzymanej wiadomości."""
        apartment_id = None
        use_alarm = hasattr(signal, "SIGALRM")  # Windows: False, Linux: True

        try:
            # Parsuj wiadomość JSON
            message_data = json.loads(body.decode("utf-8"))
            apartment_id = message_data.get("apartment_id")
            logger.info(f"Otrzymano wiadomość: {message_data}")

            if use_alarm:
                signal.signal(signal.SIGALRM, _alarm_handler)
                signal.alarm(25 * 60)

            # Przetwórz wiadomość
            try:
                processed_data = self.process_message(message_data)
            finally:
                if use_alarm:
                    signal.alarm(0)

            # ACK po sukcesie (inaczej wiadomości będą wracać lub wisieć jako unacked)
            ch.basic_ack(delivery_tag=method.delivery_tag)
            logger.info("Wiadomość przetworzona")

        except ProcessingTimeout:
            logger.warning(
                "TIMEOUT processing apartment_id=%s after %ss -> SKIP (ACK). body_tail=%s",
                apartment_id, 25 * 60, body[:500]
            )
            ch.basic_ack(delivery_tag=method.delivery_tag)

        except json.JSONDecodeError as e:
            logger.error(f"Błędny format JSON: {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

        except Exception as e:
            logger.error(f"Błąd przetwarzania: {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    def start(self):
        """Rozpocznij nasłuchiwanie wiadomości."""
        try:
            self.channel.basic_consume(
                queue=INPUT_QUEUE,
                on_message_callback=self.on_message
            )

            logger.info(f"Oczekiwanie na wiadomości z kolejki '{INPUT_QUEUE}'. CTRL+C aby zakończyć")
            self.channel.start_consuming()
        except KeyboardInterrupt:
            logger.info("Zatrzymywanie...")
            self.stop()

    def stop(self):
        """Zatrzymaj i zamknij połączenie."""
        if self.channel and not self.channel.is_closed:
            self.channel.stop_consuming()
        if self.connection and not self.connection.is_closed:
            self.connection.close()
        logger.info("Połączenie zamknięte")


def signal_handler(signum, frame):
    """Obsługa sygnału zakończenia."""
    logger.info("Otrzymano sygnał zakończenia")
    sys.exit(0)


def main():
    """Główna funkcja aplikacji."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    processor = RabbitMQProcessor()

    try:
        processor.connect()
        processor.start()
    except Exception as e:
        logger.error(f"Błąd aplikacji: {e}")
        sys.exit(1)
    finally:
        processor.stop()


if __name__ == '__main__':
    main()
