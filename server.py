from flask import Flask, request, jsonify
import mysql.connector
import requests  # 키오스크로 상품 정보를 보내기 위한 라이브러리

app = Flask(__name__)

# MySQL 데이터베이스 연결 설정
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'psh0811',
    'database': 'snack'
}

# 키오스크 서버 주소 (키오스크 컴퓨터 IP와 포트를 지정)
KIOSK_SERVER_URL = 'http://127.0.0.1:5001/receive_product'

# MySQL 데이터베이스 연결 함수
def get_db_connection():
    conn = mysql.connector.connect(
        host=db_config['host'],
        user=db_config['user'],
        password=db_config['password'],
        database=db_config['database']
    )
    return conn

# 상품 번호로 데이터베이스에서 상품 정보 조회
def get_product_info(product_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    query = "SELECT product_name, product_price, product_image FROM info WHERE product_id = %s"
    cursor.execute(query, (product_id,))
    product = cursor.fetchone()

    cursor.close()
    conn.close()
    
    return product

# 상품 번호를 받아 MySQL에서 정보를 조회하고 키오스크로 전송하는 엔드포인트
@app.route('/product', methods=['POST'])
def product():
    data = request.get_json()
    product_id = data.get('product_id')

    # 상품 정보를 데이터베이스에서 가져옴
    product_info = get_product_info(product_id)

    if product_info:
        # 상품 정보를 키오스크 서버로 전송
        try:
            response = requests.post(KIOSK_SERVER_URL, json=product_info)
            if response.status_code == 200:
                print(f"키오스크로 상품 정보 전송 성공: {product_info['name']}")
            else:
                print(f"키오스크로 상품 정보 전송 실패: {response.status_code}")
        except Exception as e:
            print(f"키오스크 서버와 통신 중 오류 발생: {e}")
        
        return jsonify(product_info), 200
    else:
        return jsonify({"error": "상품을 찾을 수 없습니다."}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
