import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHARTS_DIR = os.path.join(BASE_DIR, "cafe_charts")
RAW_CSV = os.path.join(BASE_DIR, "original_cafe_sales.csv")
CLEAN_CSV = os.path.join(BASE_DIR, "cleaned_cafe_sales.csv")

os.makedirs(CHARTS_DIR, exist_ok=True)
plt.rcParams["figure.figsize"] = (9, 5)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

df_raw = None
df_clean = None


def generate_dataset(n=450):
    np.random.seed(101)
    menu_items = {
        "Espresso": ("Hot Beverages", 120.0),
        "Cappuccino": ("Hot Beverages", 160.0),
        "Masala Chai": ("Hot Beverages", 70.0),
        "Cold Brew": ("Cold Beverages", 190.0),
        "Iced Latte": ("Cold Beverages", 180.0),
        "Croissant": ("Bakery & Snacks", 140.0),
        "Paneer Sandwich": ("Bakery & Snacks", 175.0),
        "Brownie": ("Desserts", 130.0),
    }
    order_types = ["Dine-In", "Takeaway", "Delivery"]
    payment_methods = ["UPI", "Cash", "Card", "Wallet"]

    order_ids = np.arange(5001, 5001 + n)
    timestamps = pd.date_range("2026-09-01 07:30:00", periods=n, freq="4h").astype(str)
    items = np.random.choice(
        list(menu_items.keys()),
        size=n,
        p=[0.12, 0.16, 0.14, 0.11, 0.13, 0.11, 0.12, 0.11],
    )

    df = pd.DataFrame({
        "Order_ID": order_ids,
        "Order_Timestamp": timestamps,
        "Menu_Item": items,
        "Category": [menu_items[i][0] for i in items],
        "Order_Type": np.random.choice(order_types, size=n, p=[0.48, 0.30, 0.22]),
        "Payment_Method": np.random.choice(payment_methods, size=n, p=[0.55, 0.18, 0.17, 0.10]),
        "Quantity": np.random.randint(1, 5, size=n).astype(float),
        "Unit_Price": [menu_items[i][1] for i in items],
        "Prep_Time_Mins": np.round(np.random.normal(loc=9.0, scale=3.0, size=n).clip(2, 25), 1),
        "Customer_Rating": np.round(np.random.uniform(3.0, 5.0, size=n), 1),
    })

    price_str = df["Unit_Price"].astype(str)
    mask = np.random.rand(n) < 0.25
    price_str[mask] = "₹" + price_str[mask]
    df["Unit_Price"] = price_str

    for col, rate in [
        ("Quantity", 0.04),
        ("Prep_Time_Mins", 0.05),
        ("Customer_Rating", 0.05),
        ("Payment_Method", 0.03),
    ]:
        idx = np.random.choice(df.index, size=int(n * rate), replace=False)
        df.loc[idx, col] = np.nan

    df = pd.concat([df, df.iloc[:10].copy()], ignore_index=True)
    df.loc[df.index[:4], "Quantity"] = [18.0, 22.0, 16.0, 25.0]
    df.loc[df.index[5:8], "Prep_Time_Mins"] = [48.0, 55.0, 52.0]

    df.to_csv(RAW_CSV, index=False)
    print(f"[OK] Generated raw dataset: {RAW_CSV} ({df.shape[0]} rows, {df.shape[1]} cols)")
    return df


def clean_dataset(raw_df):
    df = raw_df.copy()

    # 1. Fix data types
    df["Order_Timestamp"] = pd.to_datetime(df["Order_Timestamp"], errors="coerce")
    df["Unit_Price"] = pd.to_numeric(
        df["Unit_Price"].astype(str).str.replace("₹", "", regex=False).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )
    for col in ["Quantity", "Prep_Time_Mins", "Customer_Rating"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["Menu_Item", "Category", "Order_Type", "Payment_Method"]:
        df[col] = df[col].astype(str).str.strip()
        df.loc[df[col].isin(["nan", "None", ""]), col] = np.nan
        df[col] = df[col].astype("category")

    # 2. Remove duplicates
    before_dups = len(df)
    df = df.drop_duplicates().reset_index(drop=True)

    # 3. Handle missing values
    for cat_col in ["Menu_Item", "Category", "Order_Type", "Payment_Method"]:
        if df[cat_col].isnull().any():
            mode_val = df[cat_col].mode()[0]
            if mode_val not in df[cat_col].cat.categories:
                df[cat_col] = df[cat_col].cat.add_categories([mode_val])
            df[cat_col] = df[cat_col].fillna(mode_val)

    for num_col in ["Quantity", "Unit_Price", "Prep_Time_Mins", "Customer_Rating"]:
        if df[num_col].isnull().any():
            df[num_col] = df[num_col].fillna(df[num_col].median())

    df = df.dropna(subset=["Order_ID", "Order_Timestamp"]).reset_index(drop=True)

    # 4. Cap outliers via IQR
    outlier_summary = {}
    for col in ["Quantity", "Prep_Time_Mins"]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        cnt = int(((df[col] < lower) | (df[col] > upper)).sum())
        df[col] = df[col].clip(lower=lower, upper=upper)
        outlier_summary[col] = cnt

    # 5. Feature engineering
    df["Order_Revenue"] = np.round(df["Quantity"] * df["Unit_Price"], 2)
    df["Date"] = df["Order_Timestamp"].dt.date.astype(str)
    df["Hour"] = df["Order_Timestamp"].dt.hour
    df["Shift"] = df["Hour"].apply(
        lambda h: "Morning" if 6 <= h < 12 else ("Afternoon" if 12 <= h < 17 else "Evening/Night")
    ).astype("category")

    df.to_csv(CLEAN_CSV, index=False)
    print(f"[OK] Cleaned dataset saved: {CLEAN_CSV} ({df.shape[0]} rows, {df.shape[1]} cols)")
    print(f"     Removed duplicates: {before_dups - len(df)} | Capped outliers: {outlier_summary}")
    return df


def require_clean(df):
    if df is None:
        print("[!] Run command 1 or 2 first so cleaned data is ready.")
        return False
    return True


def plot_kpi(kpi_num, show_window=True):
    if not require_clean(df_clean):
        return

    plt.figure()
    if kpi_num == 1:
        data = df_clean.groupby("Category", observed=False)["Order_Revenue"].sum().sort_values(ascending=False)
        data.plot(kind="bar", color="#6F4E37")
        plt.title("KPI 1: Total Revenue by Menu Category")
        plt.xlabel("Category")
        plt.ylabel("Revenue (₹)")
        plt.xticks(rotation=15)
        fname = "kpi1_revenue_by_category.png"
    elif kpi_num == 2:
        data = df_clean.groupby("Menu_Item", observed=False)["Quantity"].sum().sort_values(ascending=True)
        data.plot(kind="barh", color="#2E7D32")
        plt.title("KPI 2: Total Items Sold by Menu Item")
        plt.xlabel("Quantity Sold")
        plt.ylabel("Menu Item")
        fname = "kpi2_top_menu_items.png"
    elif kpi_num == 3:
        data = df_clean.groupby("Order_Type", observed=False)["Order_Revenue"].mean().sort_values(ascending=False)
        data.plot(kind="bar", color="#D84315")
        plt.title("KPI 3: Average Order Value by Order Type")
        plt.xlabel("Order Type")
        plt.ylabel("Average Ticket (₹)")
        plt.xticks(rotation=0)
        fname = "kpi3_aov_by_order_type.png"
    elif kpi_num == 4:
        data = df_clean.groupby("Shift", observed=False)["Order_Revenue"].sum().sort_values(ascending=False)
        data.plot(kind="bar", color="#1565C0")
        plt.title("KPI 4: Revenue Across Shifts")
        plt.xlabel("Shift")
        plt.ylabel("Revenue (₹)")
        plt.xticks(rotation=0)
        fname = "kpi4_revenue_by_shift.png"
    elif kpi_num == 5:
        data = df_clean["Payment_Method"].value_counts()
        data.plot(kind="pie", autopct="%1.1f%%", startangle=140)
        plt.title("KPI 5: Order Share by Payment Method")
        plt.ylabel("")
        fname = "kpi5_payment_share.png"
    elif kpi_num == 6:
        data = df_clean.groupby("Category", observed=False)["Prep_Time_Mins"].mean().sort_values(ascending=False)
        data.plot(kind="bar", color="#8E24AA")
        plt.title("KPI 6: Average Kitchen Prep Time by Category")
        plt.xlabel("Category")
        plt.ylabel("Avg Prep Time (Mins)")
        plt.xticks(rotation=15)
        fname = "kpi6_prep_time_by_category.png"
    elif kpi_num == 7:
        data = df_clean.groupby("Date")["Order_Revenue"].sum()
        data.plot(kind="line", color="#00838F", linewidth=1.8)
        plt.title("KPI 7: Daily Revenue Trend")
        plt.xlabel("Date")
        plt.ylabel("Daily Revenue (₹)")
        plt.xticks(rotation=45)
        fname = "kpi7_daily_revenue_trend.png"
    else:
        print("[!] Invalid KPI number.")
        plt.close()
        return

    plt.tight_layout()
    out_path = os.path.join(CHARTS_DIR, fname)
    plt.savefig(out_path)
    print(f"[OK] Saved chart: {out_path}")
    if show_window:
        plt.show()
    else:
        plt.close()


def print_menu():
    print("\n" + "=" * 52)
    print("   CAFE EDA NUMBER COMMAND CONTROLLER")
    print("=" * 52)
    print("  1  -> Generate raw dataset & auto-clean")
    print("  2  -> Load existing CSV & clean")
    print("  3  -> Inspect raw data (shape, nulls, duplicates)")
    print("  4  -> Inspect cleaned data (summary & KPI totals)")
    print("  5  -> Plot KPI 1: Revenue by Category")
    print("  6  -> Plot KPI 2: Items Sold by Menu Item")
    print("  7  -> Plot KPI 3: Avg Order Value by Order Type")
    print("  8  -> Plot KPI 4: Revenue by Shift")
    print("  9  -> Plot KPI 5: Payment Method Share")
    print(" 10  -> Plot KPI 6: Avg Prep Time by Category")
    print(" 11  -> Plot KPI 7: Daily Revenue Trend")
    print(" 12  -> Save ALL 7 KPI charts at once (no popups)")
    print("  0  -> Exit")
    print("=" * 52)


def main():
    global df_raw, df_clean

    while True:
        print_menu()
        choice = input("Enter command number: ").strip()

        if choice == "1":
            df_raw = generate_dataset()
            df_clean = clean_dataset(df_raw)
        elif choice == "2":
            custom_path = input(f"Enter CSV path (or press Enter for '{RAW_CSV}'): ").strip()
            path = custom_path if custom_path else RAW_CSV
            if os.path.exists(path):
                df_raw = pd.read_csv(path)
                df_clean = clean_dataset(df_raw)
            else:
                print(f"[!] File not found: {path}")
        elif choice == "3":
            if df_raw is None:
                print("[!] No raw dataset loaded yet. Run 1 or 2 first.")
            else:
                print("\n--- RAW DATA REPORT ---")
                print("Shape:", df_raw.shape)
                print("Missing values:\n", df_raw.isnull().sum())
                print("Duplicates:", df_raw.duplicated().sum())
                print(df_raw.head())
        elif choice == "4":
            if require_clean(df_clean):
                print("\n--- CLEANED DATA & KPI SUMMARY ---")
                print(f"Total Revenue:    ₹{df_clean['Order_Revenue'].sum():,.2f}")
                print(f"Unique Orders:    {df_clean['Order_ID'].nunique()}")
                print(f"Avg Order Value:  ₹{df_clean['Order_Revenue'].mean():,.2f}")
                print(f"Avg Prep Time:    {df_clean['Prep_Time_Mins'].mean():.2f} mins")
                print(f"Avg Rating:       {df_clean['Customer_Rating'].mean():.2f} / 5.00")
                print("\nNumeric Summary:\n", df_clean[["Quantity", "Unit_Price", "Prep_Time_Mins", "Customer_Rating", "Order_Revenue"]].describe())
        elif choice in {"5", "6", "7", "8", "9", "10", "11"}:
            kpi_id = int(choice) - 4
            plot_kpi(kpi_id, show_window=True)
        elif choice == "12":
            if require_clean(df_clean):
                for k in range(1, 8):
                    plot_kpi(k, show_window=False)
                print("[OK] All 7 charts saved.")
        elif choice == "0":
            print("Exiting. Good luck with your mini project!")
            break
        else:
            print("[!] Please enter a valid number from the menu.")


if __name__ == "__main__":
    main()