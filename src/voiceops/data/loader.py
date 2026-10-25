"""CSV data loader and relational joins for ZentroMart."""
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
import pandas as pd
from datetime import date
from voiceops.config import CSV_DIR
from voiceops.data.business_calendar import load_holidays, parse_date

class DataLoader:
    """Manages synthetic CSV tables and relational queries."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else CSV_DIR
        self.customers: Dict[str, Dict[str, Any]] = {}
        self.orders: Dict[str, Dict[str, Any]] = {}
        self.order_items: Dict[str, List[Dict[str, Any]]] = {}
        self.products: Dict[str, Dict[str, Any]] = {}
        self.pincodes: Dict[str, Dict[str, Any]] = {}
        self.holidays: Set[date] = set()
        self.support_logs: List[Dict[str, Any]] = []
        self._loaded = False
        self.load_all()

    def _read_csv(self, filename: str) -> List[Dict[str, Any]]:
        filepath = self.data_dir / filename
        if not filepath.exists():
            return []
        df = pd.read_csv(filepath, dtype=str, keep_default_na=False)
        rows = df.to_dict(orient="records")
        cleaned = []
        for row in rows:
            clean_row = {}
            for k, v in row.items():
                if v == "":
                    clean_row[k] = None
                elif v in ("True", "False"):
                    clean_row[k] = (v == "True")
                else:
                    clean_row[k] = v
            cleaned.append(clean_row)
        return cleaned

    def load_all(self):
        """Load all 7 data tables into memory."""
        # 1. Customers
        for c in self._read_csv("customers.csv"):
            self.customers[c["customer_id"]] = c
            
        # 2. Products
        for p in self._read_csv("products.csv"):
            self.products[p["sku"]] = p
            
        # 3. Pincodes
        for pin in self._read_csv("pincodes.csv"):
            self.pincodes[pin["pincode"]] = pin
            
        # 4. Orders
        for o in self._read_csv("orders.csv"):
            self.orders[o["order_id"]] = o
            
        # 5. Order items
        for item in self._read_csv("order_items.csv"):
            oid = item["order_id"]
            if oid not in self.order_items:
                self.order_items[oid] = []
            self.order_items[oid].append(item)
            
        # 6. Holidays
        self.holidays = load_holidays(self.data_dir / "holidays.csv")
        
        # 7. Support logs
        self.support_logs = self._read_csv("support_logs.csv")
        self._loaded = True

    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        clean_id = "".join(ch for ch in str(order_id) if ch.isdigit())
        return self.orders.get(clean_id)

    def get_order_items(self, order_id: str) -> List[Dict[str, Any]]:
        clean_id = "".join(ch for ch in str(order_id) if ch.isdigit())
        return self.order_items.get(clean_id, [])

    def get_customer(self, customer_id: str) -> Optional[Dict[str, Any]]:
        return self.customers.get(str(customer_id))

    def get_product(self, sku: str) -> Optional[Dict[str, Any]]:
        return self.products.get(str(sku).upper())

    def get_pincode_info(self, pincode: str) -> Optional[Dict[str, Any]]:
        return self.pincodes.get(str(pincode))

    def get_order_full(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Return order enriched with customer, items, and products."""
        order = self.get_order(order_id)
        if not order:
            return None
        
        full_order = dict(order)
        cust = self.get_customer(order.get("customer_id", ""))
        if cust:
            full_order["customer_name"] = cust.get("name")
            full_order["customer_first_name"] = cust.get("name", "").split()[0] if cust.get("name") else ""
            full_order["is_plus"] = cust.get("is_plus", False)
            full_order["customer_city"] = cust.get("city")
            full_order["customer_state"] = cust.get("state")
            # Mask phone for privacy
            phone = cust.get("phone", "")
            full_order["phone_masked"] = f"******{phone[-4:]}" if len(phone) >= 4 else "******"
            full_order["phone_last4"] = phone[-4:] if len(phone) >= 4 else ""
            
        raw_items = self.get_order_items(order_id)
        enriched_items = []
        for item in raw_items:
            item_copy = dict(item)
            prod = self.get_product(item.get("sku", ""))
            if prod:
                item_copy["brand"] = prod.get("brand")
                item_copy["warranty_months"] = prod.get("warranty_months")
                item_copy["assured"] = prod.get("assured")
                item_copy["fulfilled_by"] = prod.get("fulfilled_by")
            enriched_items.append(item_copy)
        full_order["items"] = enriched_items
        
        pin_info = self.get_pincode_info(order.get("delivery_pincode", ""))
        if pin_info:
            full_order["pincode_metro"] = pin_info.get("metro")
            full_order["pincode_express"] = pin_info.get("express_available")
            
        return full_order

    def find_product_by_query(self, query: str) -> Optional[Dict[str, Any]]:
        """Find product by SKU or fuzzy/substring title match."""
        q = query.strip()
        # Direct SKU check
        sku_clean = q.upper()
        if sku_clean in self.products:
            return self.products[sku_clean]
        
        # Substring in title or SKU
        q_lower = q.lower()
        for sku, prod in self.products.items():
            if sku.lower() in q_lower or q_lower in sku.lower():
                return prod
            if prod.get("title") and q_lower in prod["title"].lower():
                return prod
                
        # Word overlap
        best_match = None
        best_overlap = 0
        q_words = set(q_lower.split())
        for prod in self.products.values():
            if not prod.get("title"):
                continue
            title_words = set(prod["title"].lower().split())
            overlap = len(q_words & title_words)
            if overlap > best_overlap and overlap >= 2:
                best_overlap = overlap
                best_match = prod
        return best_match

# Singleton instance
_DATA_LOADER_INSTANCE: Optional[DataLoader] = None

def get_data_loader() -> DataLoader:
    global _DATA_LOADER_INSTANCE
    if _DATA_LOADER_INSTANCE is None:
        _DATA_LOADER_INSTANCE = DataLoader()
    return _DATA_LOADER_INSTANCE
