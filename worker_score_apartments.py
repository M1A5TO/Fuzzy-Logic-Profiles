

"""
Prosty przykład odbierania i wysyłania wiadomości do RabbitMQ.
"""

import os
import json
import logging
import signal
import sys
from dotenv import load_dotenv
import pika
import requests

import score_apartments_offline as scorer  # <-- kluczowa zmiana: bez subprocess i bez plików


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
INPUT_QUEUE = os.getenv('INPUT_QUEUE', 'scraper_new_offers')


class ProcessingTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise ProcessingTimeout()


class RabbitMQProcessor:
    """Prosty procesor wiadomości RabbitMQ."""

    def __init__(self):
        self.connection = None
        self.channel = None

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
    #  process_message (bez plików, bez subprocess)
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
        normalized = []
        for rel in poi_rels:
            if not isinstance(rel, dict):
                continue

            ttp = rel.get("time_to_poi")
            poi_obj = rel.get("poi") if isinstance(rel.get("poi"), dict) else None

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
            raise Exception(f"No valid POI relations for apartment_id={apartment_id_int} from API")

        # 4) SCORING w pamięci (BEZ plików i BEZ subprocess)
        # scorer.compute_scores_offline oczekuje: rels_by_apt: Dict[int, List[dict]]
        rels_by_apt = {apartment_id_int: normalized}
        scores = scorer.compute_scores_offline(apartment, rels_by_apt)

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
            if k in scores:
                payload[k] = scores[k]

        if not payload:
            raise Exception("No score fields found in scoring output")

        # 5) Zapisz do backendu przez PUT
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

    def api_put(self, path: str, payload: dict) -> dict:
        url = f"{API_BASE_URL}{path}"
        r = requests.put(url, json=payload, timeout=30)
        r.raise_for_status()
        return r.json() if r.content else {}

    def on_message(self, ch, method, properties, body):
        """Obsługa otrzymanej wiadomości."""
        apartment_id = None
        use_alarm = hasattr(signal, "SIGALRM")  # Windows: False, Linux: True

        try:
            message_data = json.loads(body.decode("utf-8"))
            apartment_id = message_data.get("apartment_id")
            logger.info(f"Otrzymano wiadomość: {message_data}")

            if use_alarm:
                signal.signal(signal.SIGALRM, _alarm_handler)
                signal.alarm(25 * 60)

            try:
                _ = self.process_message(message_data)
            finally:
                if use_alarm:
                    signal.alarm(0)

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
