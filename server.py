from flask import Flask, request, jsonify
import mysql.connector
import requests

app = Flask(__name__)

# MySQL 데이터베이스 연결 설정
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'psh0811',
    'database': 'snack'
}

# 키오스크 서버 주소
KIOSK_SERVER_URL = 'http://172.20.10.14:5001/receive_product'

# MySQL 데이터베이스 연결 함수
def get_db_connection():
    conn = mysql.connector.connect(
        host=db_config['host'],
        user=db_config['user'],
        password=db_config['password'],
        database=db_config['database']
    )
    return conn

# 상품 이름으로 데이터베이스에서 상품 정보 조회
def get_product_info(product_name):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    query = "SELECT product_name, product_price, product_image,weight FROM info WHERE product_name = %s"
    cursor.execute(query, (product_name,))
    product = cursor.fetchone()

    cursor.close()
    conn.close()
    
    return product

# 결제 완료 시 데이터베이스에서 재고를 감소시키는 함수
def reduce_stock(product_name, quantity):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 재고 업데이트 쿼리
    update_query = "UPDATE info SET product_count = product_count - %s WHERE product_name = %s AND product_count >= %s"
    cursor.execute(update_query, (quantity, product_name, quantity))
    
    conn.commit()
    affected_rows = cursor.rowcount  # 업데이트된 행 수 확인
    cursor.close()
    conn.close()
    
    return affected_rows > 0  # 재고 업데이트 성공 여부 반환

# 키오스크에서 결제 완료 시 호출하는 엔드포인트
@app.route('/update_stock', methods=['POST'])
def update_stock():
    data = request.get_json()
    cart_items = data.get('cart', [])
    
    # 각 상품의 재고를 감소시키기
    for item in cart_items:
        product_name = item['name']
        quantity = item['quantity']
        
        success = reduce_stock(product_name, quantity)
        
        if not success:
            return jsonify({"error": f"재고가 부족하거나 상품 업데이트에 실패했습니다: {product_name}"}), 400
    
    return jsonify({"status": "재고가 업데이트되었습니다."}), 200

# 상품 번호를 받아 MySQL에서 정보를 조회하고 키오스크로 전송하는 엔드포인트
@app.route('/product', methods=['POST'])
def product():
    data = request.get_json()
    product_name = data.get('product_name')

    # 상품 정보를 데이터베이스에서 가져옴
    product_info = get_product_info(product_name)

    if product_info:
        # 상품 정보를 키오스크 서버로 전송
        try:
            response = requests.post(KIOSK_SERVER_URL, json=product_info)
            if response.status_code == 200:
                print(f"키오스크로 상품 정보 전송 성공: {product_info['product_name']}")
            else:
                print(f"키오스크로 상품 정보 전송 실패: {response.status_code}")
        except Exception as e:
            print(f"키오스크 서버와 통신 중 오류 발생: {e}")
        
        return jsonify(product_info), 200
    else:
        return jsonify({"error": "상품을 찾을 수 없습니다."}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
